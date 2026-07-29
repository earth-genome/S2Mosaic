from dataclasses import FrozenInstanceError

import cv2
import numpy as np
import pytest

from s2mosaic.helpers import SceneMissingAssets
from s2mosaic.masking import (
    SCL_CLOUDY_CLASSES,
    OcmTuning,
    compute_masks_from_array,
    compute_masks_from_scl,
    get_masks,
    get_valid_mask,
)


class TestGetValidMask:
    """Unit tests for the no-data dilation step (cv2.dilate, MORPH_CROSS, it=4)."""

    def test_all_zero_input_returns_all_invalid(self):
        bands = np.zeros((3, 50, 50), dtype=np.uint16)
        mask = get_valid_mask(bands, dilation_count=4)
        assert mask.shape == (50, 50)
        assert mask.dtype == bool
        assert not mask.any()

    def test_all_data_input_returns_all_valid(self):
        bands = np.full((3, 50, 50), 100, dtype=np.uint16)
        mask = get_valid_mask(bands, dilation_count=4)
        assert mask.all()

    def test_pixel_is_invalid_only_when_all_bands_are_zero(self):
        bands = np.zeros((3, 10, 10), dtype=np.uint16)
        bands[0, 5, 5] = 100  # one band has data → pixel is valid
        mask = get_valid_mask(bands, dilation_count=0)
        assert mask[5, 5]
        assert not mask[0, 0]

    def test_dilation_count_zero_does_not_grow_mask(self):
        bands = np.full((3, 50, 50), 100, dtype=np.uint16)
        bands[:, 25, 25] = 0
        mask = get_valid_mask(bands, dilation_count=0)
        assert not mask[25, 25]
        # Direct neighbors still valid
        assert mask[24, 25]
        assert mask[26, 25]
        assert mask[25, 24]
        assert mask[25, 26]

    def test_dilation_grows_invalid_region_diamond(self):
        # Repeated MORPH_CROSS dilations form a diamond (Manhattan distance) region
        bands = np.full((3, 50, 50), 100, dtype=np.uint16)
        bands[:, 25, 25] = 0
        mask = get_valid_mask(bands, dilation_count=4)
        # Manhattan distance <= 4 from (25, 25) is invalid
        assert not mask[25, 25]  # dist 0
        assert not mask[21, 25]  # dist 4
        assert not mask[29, 25]
        assert not mask[25, 21]
        assert not mask[25, 29]
        assert not mask[23, 27]  # dist 4 (2+2)
        # Manhattan distance 5+ remains valid
        assert mask[20, 25]
        assert mask[30, 25]
        assert mask[22, 28]  # dist 5

    def test_returns_bool_dtype(self):
        bands = np.zeros((3, 10, 10), dtype=np.uint16)
        assert get_valid_mask(bands).dtype == bool


class TestComputeMasksFromScl:
    """SCL-based clear+valid mask logic."""

    def test_known_class_layout(self):
        # One pixel of every class 0..11 in a single row.
        scl = np.arange(12, dtype=np.uint8).reshape(1, 12)
        clear, valid = compute_masks_from_scl(scl, dilation_count=0)
        # Unsafe classes are not clear; vegetation, bare soil, water, and snow
        # remain clear. SCL no-data is invalid separately.
        for cls in range(12):
            expected_clear = cls not in SCL_CLOUDY_CLASSES
            assert bool(clear[0, cls]) == expected_clear, (
                f"class {cls} clear={clear[0, cls]}"
            )
        # No-data (class 0) is the only invalid pixel without dilation
        assert not valid[0, 0]
        for cls in range(1, 12):
            assert valid[0, cls]

    def test_cloudy_classes_match_constant(self):
        # Defensive: lock down the constant so changes are explicit
        assert SCL_CLOUDY_CLASSES == (1, 2, 3, 7, 8, 9, 10)

    def test_dilation_grows_invalid_around_no_data(self):
        # Single no-data pixel → 4-iter MORPH_CROSS dilate → diamond of invalids
        scl = np.full((50, 50), 5, dtype=np.uint8)  # all bare-soil
        scl[25, 25] = 0
        _, valid = compute_masks_from_scl(scl, dilation_count=4)
        assert not valid[25, 25]
        assert not valid[21, 25]  # Manhattan dist 4
        assert not valid[25, 21]
        assert valid[20, 25]  # dist 5 still valid
        assert valid[25, 20]

    def test_3d_input_squeezes_first_axis(self):
        # get_full_band returns (1, H, W); compute_masks_from_scl should handle that
        scl = np.array([[[0, 1, 4, 8]]], dtype=np.uint8)  # shape (1, 1, 4)
        clear, valid = compute_masks_from_scl(scl, dilation_count=0)
        assert clear.shape == (1, 4)
        np.testing.assert_array_equal(clear[0], [True, False, True, False])
        np.testing.assert_array_equal(valid[0], [False, True, True, True])

    def test_returns_bool_dtype(self):
        scl = np.zeros((10, 10), dtype=np.uint8)
        clear, valid = compute_masks_from_scl(scl)
        assert clear.dtype == bool
        assert valid.dtype == bool


class TestOcmTuning:
    """Sensitivity knobs on the OCM path."""

    @staticmethod
    def _patch_predict(monkeypatch, result):
        """Stub omnicloudmask, recording the kwargs s2mosaic passed it."""
        captured = {}

        def fake_predict_from_array(**kwargs):
            captured.update(kwargs)
            return result

        monkeypatch.setattr(
            "s2mosaic.masking.predict_from_array", fake_predict_from_array
        )
        return captured

    def test_defaults_request_argmax_not_confidence(self, monkeypatch):
        classes = np.zeros((1, 4, 4), dtype=np.uint8)
        captured = self._patch_predict(monkeypatch, classes)

        compute_masks_from_array(np.full((3, 4, 4), 100, dtype=np.uint16))

        assert captured["export_confidence"] is False

    def test_argmax_keeps_only_the_clear_class(self, monkeypatch):
        classes = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)  # (1, 1, 4)
        self._patch_predict(monkeypatch, classes)
        rgb_nir = np.full((3, 1, 4), 100, dtype=np.uint16)

        clear, _ = compute_masks_from_array(
            rgb_nir, tuning=OcmTuning(min_island_size=0, smooth_edge_size=0)
        )

        # Only the clear class survives; cloud/thin/shadow are masked.
        np.testing.assert_array_equal(clear[0], [True, False, False, False])

    def test_clear_threshold_uses_confidence_bands(self, monkeypatch):
        # (4, 1, 3) probabilities: clear prob 0.95, 0.55, 0.20.
        confidence = np.zeros((4, 1, 3), dtype=np.float32)
        confidence[0] = [[0.95, 0.55, 0.20]]
        captured = self._patch_predict(monkeypatch, confidence)
        rgb_nir = np.full((3, 1, 3), 100, dtype=np.uint16)

        clear, _ = compute_masks_from_array(
            rgb_nir,
            tuning=OcmTuning(
                clear_threshold=0.6, min_island_size=0, smooth_edge_size=0
            ),
        )

        assert captured["export_confidence"] is True
        assert captured["softmax_output"] is True
        # 0.55 wins the argmax but loses the 0.6 threshold — this is the wisp case.
        np.testing.assert_array_equal(clear[0], [True, False, False])

    def test_higher_threshold_is_monotonically_stricter(self, monkeypatch):
        confidence = np.zeros((4, 1, 4), dtype=np.float32)
        confidence[0] = [[0.99, 0.80, 0.60, 0.40]]
        self._patch_predict(monkeypatch, confidence)
        rgb_nir = np.full((3, 1, 4), 100, dtype=np.uint16)

        counts = []
        for threshold in (0.3, 0.5, 0.7, 0.9):
            clear, _ = compute_masks_from_array(
                rgb_nir,
                tuning=OcmTuning(
                    clear_threshold=threshold,
                    min_island_size=0,
                    smooth_edge_size=0,
                ),
            )
            counts.append(int(clear.sum()))

        assert counts == sorted(counts, reverse=True)
        assert counts == [4, 3, 2, 1]

    def test_cloud_dilation_buffers_detected_cloud(self, monkeypatch):
        classes = np.zeros((1, 11, 11), dtype=np.uint8)
        classes[0, 5, 5] = 1  # one thick-cloud pixel in a clear scene
        self._patch_predict(monkeypatch, classes)
        rgb_nir = np.full((3, 11, 11), 100, dtype=np.uint16)
        tuning = OcmTuning(cloud_dilation=2, min_island_size=0, smooth_edge_size=0)

        clear, _ = compute_masks_from_array(rgb_nir, tuning=tuning)

        # Repeated MORPH_CROSS dilations grow a Manhattan-distance diamond.
        assert not clear[5, 5]
        assert not clear[3, 5]  # dist 2
        assert not clear[4, 6]  # dist 2
        assert clear[2, 5]  # dist 3 still clear
        assert clear[0, 0]

    def test_cloud_dilation_does_not_grow_from_no_data(self, monkeypatch):
        classes = np.zeros((1, 11, 11), dtype=np.uint8)
        self._patch_predict(monkeypatch, classes)
        # A no-data hole reads as not-clear, but must not seed cloud dilation.
        rgb_nir = np.full((3, 11, 11), 100, dtype=np.uint16)
        rgb_nir[:, 5, 5] = 0
        tuning = OcmTuning(cloud_dilation=3, min_island_size=0, smooth_edge_size=0)

        clear, valid = compute_masks_from_array(rgb_nir, tuning=tuning)

        assert not valid[5, 5]
        # Neighbours outside the dilated no-data diamond keep their clear status.
        assert clear[0, 0]
        assert clear[10, 10]

    def test_cleanup_disabled_keeps_single_pixel_cloud(self, monkeypatch):
        classes = np.zeros((1, 40, 40), dtype=np.uint8)
        classes[0, 20, 20] = 2  # lone thin-cloud speck, smaller than min_island_size
        self._patch_predict(monkeypatch, classes)
        rgb_nir = np.full((3, 40, 40), 100, dtype=np.uint16)

        kept, _ = compute_masks_from_array(
            rgb_nir, tuning=OcmTuning(min_island_size=0, smooth_edge_size=0)
        )
        cleaned, _ = compute_masks_from_array(rgb_nir, tuning=OcmTuning())

        assert not kept[20, 20]
        # Default cleanup fills wisp-scale holes in the clear mask back to clear.
        assert cleaned[20, 20]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"clear_threshold": 0.0},
            {"clear_threshold": 1.0},
            {"clear_threshold": -0.1},
            {"clear_threshold": 1.5},
            {"cloud_dilation": -1},
            {"min_island_size": -1},
            {"smooth_edge_size": -1},
        ],
    )
    def test_validate_rejects_out_of_range(self, kwargs):
        with pytest.raises(ValueError):
            OcmTuning(**kwargs).validate()

    def test_validate_accepts_defaults_and_typical_values(self):
        OcmTuning().validate()
        OcmTuning(clear_threshold=0.6, cloud_dilation=1, min_island_size=2).validate()

    def test_is_hashable_and_frozen(self):
        tuning = OcmTuning(clear_threshold=0.6)
        assert hash(tuning) == hash(OcmTuning(clear_threshold=0.6))
        with pytest.raises(FrozenInstanceError):
            tuning.clear_threshold = 0.7


class TestZoomAlignmentConvention:
    """Regression test for the nearest-neighbor resample alignment.

    s2mosaic deliberately uses cv2.INTER_NEAREST (pixel-area-cell convention)
    instead of scipy.ndimage.zoom order=0 (centered-point convention) so an
    upsampled 60m S2 pixel maps exactly to the 6x6 block of 10m pixels it
    covers in the GeoTIFF transform. Reverting to scipy would shift 60m
    bands by ~30m (3 pixels at 10m).
    """

    def test_2x_upsample_matches_np_repeat(self):
        src = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        out = cv2.resize(src, (6, 4), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(out, src.repeat(2, axis=0).repeat(2, axis=1))

    def test_6x_upsample_matches_np_repeat(self):
        src = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        out = cv2.resize(src, (12, 12), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(out, src.repeat(6, axis=0).repeat(6, axis=1))

    def test_3x_upsample_uses_pixel_area_layout(self):
        # Pixel-area: input pixel i fills output cells [i*N, (i+1)*N).
        # Scipy's order=0 would yield [1, 1, 2, 2, 2, 2, 3, 3, 3] — different.
        src = np.array([[1, 2, 3]], dtype=np.uint16)
        out = cv2.resize(src, (9, 1), interpolation=cv2.INTER_NEAREST)
        expected = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=np.uint16)
        np.testing.assert_array_equal(out, expected)


class TestMaskingHelpers:
    def test_get_masks_resizes_to_rectangular_target(self, monkeypatch):
        class FakeAsset:
            href = "remote.tif"

        class FakeItem:
            assets = {"B04": FakeAsset(), "B03": FakeAsset(), "B8A": FakeAsset()}

        class FakeSource:
            def asset_name(self, canonical):
                return canonical

        seen_assets = set()
        band_values = {"B04": 4, "B03": 3, "B8A": 8}

        def fake_get_full_band(href, *, source, res, asset_name):
            seen_assets.add(asset_name)
            return np.full((1, 2, 2), band_values[asset_name], dtype=np.uint16), {}

        def fake_compute_masks_from_array(array, batch_size, inference_dtype, tuning):
            np.testing.assert_array_equal(array[:, 0, 0], np.array([4, 3, 8]))
            return (
                np.array([[True, False], [False, True]]),
                np.ones((2, 2), dtype=bool),
            )

        monkeypatch.setattr("s2mosaic.masking.get_full_band", fake_get_full_band)
        monkeypatch.setattr(
            "s2mosaic.masking.compute_masks_from_array",
            fake_compute_masks_from_array,
        )

        clear, valid = get_masks(FakeItem(), FakeSource(), target_size=(3, 5))

        assert clear.shape == (3, 5)
        assert valid.shape == (3, 5)
        assert seen_assets == {"B04", "B03", "B8A"}

    def test_get_masks_raises_when_ocm_band_missing(self):
        class FakeItem:
            assets = {"B03": object(), "B8A": object()}

        class FakeSource:
            def asset_name(self, canonical):
                return canonical

        with pytest.raises(SceneMissingAssets, match="missing STAC assets: B04"):
            get_masks(FakeItem(), FakeSource())
