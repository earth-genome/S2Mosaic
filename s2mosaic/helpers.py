import functools
import logging
import random
import re
import sys
import time
from datetime import date, datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from dateutil.relativedelta import relativedelta
from rasterio.errors import RasterioIOError
from urllib3.exceptions import HTTPError

if TYPE_CHECKING:
    from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

T = TypeVar("T")


def first_stop_reached(
    good_pixel_tracker: "npt.NDArray[Any]",
    coverage_mask: "npt.NDArray[Any]",
    first_coverage_target: Optional[float] = None,
) -> Tuple[bool, float]:
    """Whether ``first`` may stop fetching masks, and the coverage reached.

    With no target the stop requires *every* in-coverage pixel to be filled.
    Real scenes rarely clear that bar — a handful of pixels stay cloudy in
    every candidate — so the stop almost never fires and ``first`` walks the
    whole ranked candidate list. A ``first_coverage_target`` in (0, 1] stops
    once that fraction of in-coverage pixels is filled, trading a little
    nodata for skipped cloud-mask fetches and scene reads.

    Both masks must be boolean and share a shape.
    """
    coverable = int(coverage_mask.sum())
    if coverable == 0:
        return True, 1.0
    filled = int(np.count_nonzero(good_pixel_tracker & coverage_mask))
    fraction = filled / coverable
    if first_coverage_target is None:
        return filled == coverable, fraction
    return fraction >= first_coverage_target, fraction


class SceneFetchError(Exception):
    """Raised when a per-scene step cannot proceed for a recoverable reason.

    Caught at the pipeline-loop level so one bad scene doesn't abort the whole
    mosaic — the loop logs and skips. Subclasses cover exhausted fetch retries
    (:class:`SceneMissingAssets`) and scenes with no footprint overlap
    (:class:`SceneNoOverlap`). Errors that aren't scene-level recoverable (e.g.
    OCM inference failures on otherwise-valid data) bypass this and propagate.
    """


class SceneMissingAssets(SceneFetchError):
    """Scene STAC item lacks assets required for cloud masking or compositing."""


class SceneNoOverlap(SceneFetchError):
    """Subclass for scenes whose footprint doesn't intersect ``bounds_target``.

    Not a failure — these are returned by the STAC search (which queries a
    slightly inflated lat/lng envelope to cover the UTM output extent) but
    have no pixels in the requested area. The pipeline silently drops them
    without counting them as dropped_scenes or logging a warning.
    """


def _exception_chain_summary(exc: BaseException) -> str:
    """Compactly format an exception plus its Python cause/context chain."""
    parts = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


def report_dropped_scenes(
    dropped: List[Dict[str, str]],
    *,
    total: int,
    stream: Any = None,
) -> None:
    """Emit a single end-of-phase line listing scenes that fell out of Phase 1.

    Goes through ``print`` (defaulting to ``stderr``) so it's visible even when
    the user hasn't configured Python logging — the warning logs inside the
    retry loop only surface if a handler is attached. ``stream`` is an
    injection point for tests; in production we always write to ``stderr``.
    """
    if not dropped:
        return
    handle = stream if stream is not None else sys.stderr
    ids = ", ".join(d["id"] for d in dropped)
    print(
        f"s2mosaic: {len(dropped)}/{total} scenes dropped due to fetch "
        f"errors after retries: {ids}",
        file=handle,
    )


def backoff_delay(
    attempt: int,
    *,
    base: float = 0.5,
    factor: float = 2.0,
    cap: float = 8.0,
    jitter: float = 0.25,
) -> float:
    """Exponential backoff with symmetric jitter, capped at ``cap`` seconds.

    Shared by the per-scene fetch decorator (Phase 1) and the per-tile reader
    retry loops (Phase 2). ``attempt`` is 0-indexed (delay *after* attempt 0
    is ``base``). Jitter is a fraction (e.g. ``0.25`` means ±25%) applied
    multiplicatively after the cap, so the realised delay can briefly exceed
    ``cap`` by up to ``cap*jitter``. Cap and jitter together prevent runaway
    delays and thundering-herd when many threads back off at the same exponent.
    """
    raw = base * (factor**attempt)
    bounded = min(raw, cap)
    if jitter > 0:
        bounded *= 1.0 + random.uniform(-jitter, jitter)
    return max(0.0, bounded)


def _is_retryable_exception(
    exc: BaseException, retry_exceptions: Tuple[type[BaseException], ...]
) -> bool:
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, retry_exceptions):
            return True
        current = current.__cause__ or current.__context__
    return False


def with_scene_retry(
    attempts: int = 3,
    base_delay: float = 1.0,
    retry_exceptions: Tuple[type[BaseException], ...] = (
        RasterioIOError,
        OSError,
        TimeoutError,
        HTTPError,
    ),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry a per-scene fetcher with exponential backoff + jitter.

    On exhaustion, the last exception is wrapped in :class:`SceneFetchError`
    so the pipeline loop can catch fetch failures specifically without also
    swallowing inference or programming errors that arise outside the fetch.
    Backoff is delegated to :func:`backoff_delay` (shared with the per-tile
    reader retries) and seeded with ``base_delay``.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[BaseException] = None
            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if not _is_retryable_exception(e, retry_exceptions):
                        raise
                    last_exc = e
                    if attempt < attempts - 1:
                        delay = backoff_delay(attempt, base=base_delay)
                        logger.warning(
                            f"{fn.__name__} attempt {attempt + 1}/{attempts} "
                            f"failed: {_exception_chain_summary(e)}; "
                            f"retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
            assert last_exc is not None
            raise SceneFetchError(
                f"{fn.__name__} failed after {attempts} attempts: "
                f"{_exception_chain_summary(last_exc)}"
            ) from last_exc

        return wrapper

    return decorator


def get_band_template(
    bands: List[str],
) -> Tuple[List[Tuple[str, int]], int, List[int]]:
    """Return per-band STAC asset/raster-band template + sizes.

    The result is the same shape for grid_id and bounds modes:

        * ``href_template`` — list of ``(stac_asset_name, raster_band_idx)``;
          one entry per output band.
        * ``bands_count`` — number of output bands.
        * ``href_band_indices`` — just the raster band indices, pulled out
          for the hot path.

    ``"visual"`` is the 3-band TCI asset; spectral requests are one asset
    per band, each reading raster band 1.
    """
    is_visual = "visual" in bands
    if is_visual:
        href_template: List[Tuple[str, int]] = [
            ("visual", 1),
            ("visual", 2),
            ("visual", 3),
        ]
        bands_count = 3
    else:
        href_template = [(band, 1) for band in bands]
        bands_count = len(bands)
    href_band_indices = [band_idx for _, band_idx in href_template]
    return href_template, bands_count, href_band_indices


def get_rasterio_resampling(method: str) -> "Resampling":
    """Map a string resampling method to a rasterio.enums.Resampling value."""
    from rasterio.enums import Resampling

    return {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "lanczos": Resampling.lanczos,
    }[method]


OCM_MIN_RESOLUTION = 20
OCM_MAX_RESOLUTION = 50
OCM_MASK_BANDS: Tuple[str, ...] = ("B04", "B03", "B8A")

# MGRS tile is exactly 109800m on each side.
MGRS_TILE_SIZE_M = 109800


def missing_item_assets(
    item: Any,
    source: Any,
    canonical_bands: List[str],
) -> List[str]:
    """Return provider STAC asset keys absent from ``item``."""
    assets = getattr(item, "assets", None)
    if assets is None:
        return [source.asset_name(band) for band in canonical_bands]
    missing: List[str] = []
    for band in canonical_bands:
        asset_key = source.asset_name(band)
        if asset_key not in assets:
            missing.append(asset_key)
    return missing


def required_assets_for_mosaic(bands: List[str], cloud_mask: str) -> List[str]:
    """Canonical band/asset names that must exist on every composited scene."""
    from .config import CLOUD_MASK_OCM, CLOUD_MASK_SCL

    required = set(bands)
    if cloud_mask == CLOUD_MASK_OCM:
        required.update(OCM_MASK_BANDS)
    elif cloud_mask == CLOUD_MASK_SCL:
        required.add("SCL")
    return sorted(required)


def ensure_item_assets(
    item: Any,
    source: Any,
    canonical_bands: List[str],
) -> None:
    """Raise :class:`SceneMissingAssets` when ``item`` lacks required assets."""
    missing = missing_item_assets(item, source, canonical_bands)
    if missing:
        raise SceneMissingAssets(f"missing STAC assets: {', '.join(missing)}")


def partition_items_by_required_assets(
    items: List[Any],
    source: Any,
    bands: List[str],
    cloud_mask: str,
) -> Tuple[List[Any], List[Dict[str, str]]]:
    """Drop scenes whose STAC items lack assets for ``bands`` / ``cloud_mask``."""
    required = required_assets_for_mosaic(bands, cloud_mask)
    kept: List[Any] = []
    dropped: List[Dict[str, str]] = []
    for item in items:
        missing = missing_item_assets(item, source, required)
        if missing:
            dropped.append(
                {
                    "id": item.id,
                    "reason": f"missing STAC assets: {', '.join(missing)}",
                }
            )
        else:
            kept.append(item)
    return kept, dropped


def filter_scene_dataframe(
    scenes_df: Any,
    source: Any,
    bands: List[str],
    cloud_mask: str,
) -> Tuple[Any, List[Dict[str, str]]]:
    """Return a scenes DataFrame with incomplete STAC items removed."""
    import pandas as pd

    from .stac import ITEM_COL

    kept_rows = []
    dropped: List[Dict[str, str]] = []
    required = required_assets_for_mosaic(bands, cloud_mask)
    for _, row in scenes_df.iterrows():
        item = row[ITEM_COL]
        missing = missing_item_assets(item, source, required)
        if missing:
            dropped.append(
                {
                    "id": item.id,
                    "reason": f"missing STAC assets: {', '.join(missing)}",
                }
            )
        else:
            kept_rows.append(row)
    if not kept_rows:
        return scenes_df.iloc[0:0].copy(), dropped
    return pd.DataFrame(kept_rows).reset_index(drop=True), dropped


def pick_ocm_resolution(user_resolution: int) -> int:
    """Pick the OCM input resolution given the user's output resolution.

    OCM is fastest at coarser resolutions (less data to transfer / process).
    20m is the recommended default; 50m is the coarsest tested. We use the
    user's resolution where it falls in [20, 50], and clamp otherwise:
        user <= 20 → 20  (e.g. 10m output keeps OCM at 20m)
        20 < user < 50 → user
        user >= 50 → 50
    """
    return max(OCM_MIN_RESOLUTION, min(user_resolution, OCM_MAX_RESOLUTION))


# MGRS Sentinel-2 grid id pattern.
#
# Three pieces:
#   - UTM zone 1-60 (no leading zero — both MPC and Element 84 use bare ints)
#   - latitude band letter: C-X excluding I and O (A, B, Y, Z are polar UPS;
#     I and O are skipped to avoid confusion with 1 and 0)
#   - 100 km grid square: column letter A-Z excluding I and O (24 letters),
#     row letter A-V excluding I and O (20 letters)
#
# Accepts both 4-char (zones 1-9) and 5-char (zones 10-60) ids.
GRID_ID_PATTERN = re.compile(
    r"^(?:[1-9]|[1-5][0-9]|60)[C-HJ-NP-X][A-HJ-NP-Z][A-HJ-NP-V]$"
)


def normalize_grid_id(grid_id: str) -> str:
    """Normalize and validate a Sentinel-2 MGRS grid id.

    Strips whitespace, uppercases the input, and matches it against
    :data:`GRID_ID_PATTERN`. Returns the normalized id on success.

    Raises ``ValueError`` if the id is empty or doesn't match the MGRS
    grid-square format used by Sentinel-2 (e.g. ``'50HMK'`` or ``'1FBE'``).
    """
    if not isinstance(grid_id, str):
        raise ValueError(f"grid_id must be a string, got {type(grid_id).__name__}")
    normalized = grid_id.strip().upper()
    if not normalized:
        raise ValueError("grid_id must not be empty")
    if not GRID_ID_PATTERN.match(normalized):
        raise ValueError(
            f"grid_id {grid_id!r} is not a valid Sentinel-2 MGRS tile id. "
            "Expected format like '50HMK' (UTM zone 1-60, latitude band "
            "C-X excluding I/O, then two grid-square letters). See "
            "https://sentiwiki.copernicus.eu/web/s2-products and "
            "https://dpird-dma.github.io/Sentinel-2-grid-explorer/"
        )
    return normalized


def define_dates(
    start_year: int,
    start_month: int,
    start_day: int,
    duration_years: int,
    duration_months: int,
    duration_days: int,
) -> Tuple[date, date]:
    start_date = datetime(start_year, start_month, start_day)
    end_date = start_date + relativedelta(
        years=duration_years, months=duration_months, days=duration_days
    )
    return start_date, end_date
