import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import pystac
from multiclean import clean_array
from omnicloudmask import predict_from_array

from .data_reader import get_full_band
from .helpers import ensure_item_assets
from .sources import Source

# OmniCloudMask output classes.
OCM_CLASS_CLEAR: int = 0
OCM_CLASS_THICK_CLOUD: int = 1
OCM_CLASS_THIN_CLOUD: int = 2
OCM_CLASS_CLOUD_SHADOW: int = 3


@dataclass(frozen=True)
class OcmTuning:
    """Sensitivity controls for the OCM cloud-mask path.

    The defaults reproduce the original behaviour exactly: hard argmax over
    the four OCM classes, island/edge cleanup of the clear mask, and no cloud
    buffer.

    Args:
        clear_threshold: When set, a pixel counts as clear only if its softmax
            probability for the clear class is >= this value, rather than
            merely winning the argmax. Values above ~0.5 make the mask
            progressively more suspicious, which is the lever for thin cirrus
            and cloud wisps that only narrowly beat the cloud classes. ``None``
            keeps argmax.
        cloud_dilation: Number of 3x3 cross dilations applied to the detected
            cloud mask before it is subtracted from clear. Buffers the wisps
            and haloes that fringe a confident detection. Counted in pixels at
            the OCM working resolution, which is coarser than the output
            resolution.
        min_island_size: Minimum connected-component size kept during clear
            mask cleanup. This cleanup runs on the *clear* mask, so it removes
            small *cloud* detections (holes in clear); lowering it preserves
            wisp-scale detections.
        smooth_edge_size: Circular kernel size for clear mask edge smoothing.
            Set together with ``min_island_size`` to 0 to skip cleanup.
    """

    clear_threshold: Optional[float] = None
    cloud_dilation: int = 0
    min_island_size: int = 8
    smooth_edge_size: int = 3

    def validate(self) -> None:
        if self.clear_threshold is not None and not 0.0 < self.clear_threshold < 1.0:
            raise ValueError(
                "OcmTuning.clear_threshold must be in (0, 1), got "
                f"{self.clear_threshold}"
            )
        for name in ("cloud_dilation", "min_island_size", "smooth_edge_size"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"OcmTuning.{name} must be >= 0, got {value}")


DEFAULT_OCM_TUNING = OcmTuning()

# Sentinel-2 SCL band class values:
#   0  no_data       6  water
#   1  saturated     7  unclassified
#   2  dark/shadow   8  cloud_medium_probability
#   3  cloud_shadow  9  cloud_high_probability
#   4  vegetation   10  thin_cirrus
#   5  bare_soil    11  snow
# Treated as unsafe for "clear": saturated, dark/shadow, cloud shadow,
# unclassified, both cloud probabilities, thin cirrus.
SCL_CLOUDY_CLASSES: Tuple[int, ...] = (1, 2, 3, 7, 8, 9, 10)
SCL_NO_DATA: int = 0


def _dilate_no_data(no_data: npt.NDArray[Any], dilation_count: int) -> npt.NDArray[Any]:
    """Dilate a no-data mask (1=no_data) by `dilation_count` cross-3x3 iterations."""
    if dilation_count <= 0:
        return no_data
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.dilate(no_data, kernel, iterations=dilation_count)


def _buffer_cloud(
    clear: npt.NDArray[Any], valid: npt.NDArray[Any], dilation_count: int
) -> npt.NDArray[Any]:
    """Grow detected cloud into neighbouring clear pixels.

    Dilation is seeded from cloud *within the scene footprint* so that
    scene-edge no-data (which OCM zeroes, and which therefore reads as
    not-clear) cannot eat into real observations.
    """
    if dilation_count <= 0:
        return clear
    cloud = (valid & ~clear).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cloud = cv2.dilate(cloud, kernel, iterations=dilation_count)
    return clear & (cloud == 0)  # type: ignore[no-any-return, unused-ignore]


def get_valid_mask(
    bands: npt.NDArray[Any], dilation_count: int = 4
) -> npt.NDArray[Any]:
    # create mask to remove pixels with no data, add dilation to remove edge pixels
    no_data = (bands.sum(axis=0) == 0).astype(np.uint8)
    no_data = _dilate_no_data(no_data, dilation_count)
    return no_data == 0  # type: ignore[no-any-return, unused-ignore]


def compute_masks_from_scl(
    scl: npt.NDArray[Any], dilation_count: int = 4
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Build (clear, valid) masks from an SCL band.

    Mirrors :func:`compute_masks_from_array` so OCM and SCL providers are
    interchangeable. ``clear`` is True where the pixel's SCL class is safe for
    compositing; ``valid`` is True where SCL != 0, dilated to erode scene-edge
    no-data the same way the OCM path does.
    """
    if scl.ndim == 3 and scl.shape[0] == 1:
        scl = scl[0]
    no_data = (scl == SCL_NO_DATA).astype(np.uint8)
    no_data = _dilate_no_data(no_data, dilation_count)
    valid = no_data == 0
    clear = ~np.isin(scl, SCL_CLOUDY_CLASSES)
    return clear, valid


def compute_masks_from_array(
    rgb_nir: npt.NDArray[Any],
    batch_size: int = 6,
    inference_dtype: str = "fp32",
    tuning: Optional[OcmTuning] = None,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Run cloud + valid masking on an in-memory (3, H, W) R+G+NIR uint16 array.

    Returns (clear_mask, valid_mask) at the same resolution as the input.
    ``tuning`` controls mask sensitivity; see :class:`OcmTuning`.

    Suppresses omnicloudmask's "Significant no-data areas detected" warning —
    it fires on every cross-UTM-zone edge scene where the swath polygon is
    tilted relative to the read rectangle (triangular nodata wedge). OCM
    auto-shrinks the patch size and produces correct masks; the warning is
    just noise.
    """
    tuning = tuning if tuning is not None else DEFAULT_OCM_TUNING
    use_confidence = tuning.clear_threshold is not None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Significant no-data areas detected",
            category=UserWarning,
        )
        patch_size = min(*rgb_nir.shape[1:], 1000)
        patch_overlap = min(patch_size // 2, 50)
        prediction = predict_from_array(
            input_array=rgb_nir,
            batch_size=batch_size,
            inference_dtype=inference_dtype,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            export_confidence=use_confidence,
            softmax_output=True,
        )
    if use_confidence:
        # Confidence output is (4, H, W) of per-class probabilities. No-data
        # pixels are zeroed by OCM, so they read as not-clear and are removed
        # by ``valid`` below.
        clear_prob = prediction[OCM_CLASS_CLEAR]
        clear = (clear_prob >= tuning.clear_threshold).astype(np.uint8)
    else:
        clear = (prediction[0] == OCM_CLASS_CLEAR).astype(np.uint8)
    if tuning.min_island_size > 0 or tuning.smooth_edge_size > 0:
        clear = clean_array(
            clear,
            min_island_size=tuning.min_island_size,
            smooth_edge_size=tuning.smooth_edge_size,
            connectivity=4,
        )
    clear_bool = clear.astype(bool)
    valid = get_valid_mask(rgb_nir)
    clear_bool = _buffer_cloud(clear_bool, valid, tuning.cloud_dilation)
    return clear_bool, valid


def get_scl_masks(
    item: pystac.Item,
    source: Source,
    user_resolution: int = 10,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """SCL-based clear+valid masks at the user's output resolution.

    Cheaper than :func:`get_masks` (one COG read, no DL inference) but less
    accurate — relies on the L2A processor's published Scene Classification
    Layer rather than re-running cloud detection.
    """
    ensure_item_assets(item, source, ["SCL"])
    href = item.assets[source.asset_name("SCL")].href
    arr, _ = get_full_band(
        href=href, source=source, res=user_resolution, asset_name="SCL"
    )
    return compute_masks_from_scl(arr)


def get_masks(
    item: pystac.Item,
    source: Source,
    batch_size: int = 6,
    inference_dtype: str = "fp32",
    max_dl_workers: int = 4,
    target_size: Union[int, Tuple[int, int]] = 10980,
    ocm_resolution: int = 20,
    tuning: Optional[OcmTuning] = None,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    # download RG+NIR bands at OCM resolution for cloud masking
    ocm_bands = ["B04", "B03", "B8A"]
    ensure_item_assets(item, source, ocm_bands)

    def get_band_at_ocm_res(
        band: str,
    ) -> Tuple[npt.NDArray[np.uint16], dict[str, Any]]:
        return get_full_band(
            href=item.assets[source.asset_name(band)].href,
            source=source,
            res=ocm_resolution,
            asset_name=band,
        )

    with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
        bands_and_profiles = list(executor.map(get_band_at_ocm_res, ocm_bands))

    band_arrays, _ = zip(*bands_and_profiles, strict=False)
    ocm_input = np.vstack(band_arrays)

    clear, valid = compute_masks_from_array(
        ocm_input,
        batch_size=batch_size,
        inference_dtype=inference_dtype,
        tuning=tuning,
    )
    # Resample masks from OCM resolution (20m) to the target output shape.
    target_height, target_width = (
        (target_size, target_size) if isinstance(target_size, int) else target_size
    )
    if clear.shape != (target_height, target_width):
        clear = cv2.resize(
            clear.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        valid = cv2.resize(
            valid.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return clear, valid
