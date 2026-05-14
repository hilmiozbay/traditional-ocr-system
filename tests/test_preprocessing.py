"""
tests/test_preprocessing.py
============================
preprocessing.py modülü için birim testler (pytest).
"""

import numpy as np
import pytest
import cv2

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from preprocessing import (
    resize_image,
    denoise,
    enhance_contrast,
    binarize,
    detect_skew_hough,
    detect_skew_projection,
    deskew,
    morphological_clean,
    preprocess_pipeline,
)


# ─── Yardımcı ──────────────────────────────────

def make_white_bgr(h=400, w=300) -> np.ndarray:
    """Tamamen beyaz BGR görüntüsü üretir."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def make_text_binary(h=200, w=400) -> np.ndarray:
    """Yatay metin çizgileri olan sahte binary görüntü."""
    img = np.full((h, w), 255, dtype=np.uint8)
    for y in [30, 60, 90, 120, 150]:
        img[y:y+8, 20:380] = 0  # siyah çizgiler
    return img


# ─── resize_image ──────────────────────────────

class TestResizeImage:
    def test_no_resize_needed(self):
        img = make_white_bgr(200, 100)
        out = resize_image(img, max_width=3000, max_height=4000)
        assert out.shape == img.shape

    def test_resize_width(self):
        img = make_white_bgr(100, 5000)
        out = resize_image(img, max_width=1000, max_height=4000)
        assert out.shape[1] <= 1000

    def test_resize_height(self):
        img = make_white_bgr(6000, 100)
        out = resize_image(img, max_width=3000, max_height=1000)
        assert out.shape[0] <= 1000

    def test_aspect_ratio_preserved(self):
        img = make_white_bgr(400, 200)
        out = resize_image(img, max_width=100, max_height=4000)
        orig_ratio = img.shape[1] / img.shape[0]
        out_ratio = out.shape[1] / out.shape[0]
        assert abs(orig_ratio - out_ratio) < 0.05


# ─── denoise ───────────────────────────────────

class TestDenoise:
    @pytest.mark.parametrize("method", ["gaussian", "median", "bilateral"])
    def test_output_shape(self, method):
        img = make_white_bgr()
        out = denoise(img, method=method)
        assert out.shape == img.shape

    def test_unknown_method_returns_original(self):
        img = make_white_bgr()
        out = denoise(img, method="nonexistent")
        np.testing.assert_array_equal(out, img)

    def test_gaussian_kernel_even_auto_fixed(self):
        img = make_white_bgr()
        # çift çekirdek boyutu otomatik düzeltilmeli
        out = denoise(img, method="gaussian", gaussian_kernel=4)
        assert out.shape == img.shape


# ─── enhance_contrast ──────────────────────────

class TestEnhanceContrast:
    def test_clahe_output_2d(self):
        img = make_white_bgr()
        out = enhance_contrast(img, method="clahe")
        assert out.ndim == 2

    def test_histogram_eq_output_2d(self):
        img = make_white_bgr()
        out = enhance_contrast(img, method="histogram_eq")
        assert out.ndim == 2

    def test_none_returns_input(self):
        img = make_white_bgr()
        out = enhance_contrast(img, method="none")
        np.testing.assert_array_equal(out, img)


# ─── binarize ──────────────────────────────────

class TestBinarize:
    @pytest.mark.parametrize("method", ["otsu", "adaptive_mean", "adaptive_gaussian"])
    def test_binary_values(self, method):
        img = make_white_bgr()
        out = binarize(img, method=method)
        unique_vals = np.unique(out)
        assert set(unique_vals).issubset({0, 255})

    def test_output_shape_2d(self):
        img = make_white_bgr()
        out = binarize(img, method="otsu")
        assert out.ndim == 2


# ─── deskew ────────────────────────────────────

class TestDeskew:
    def test_deskew_returns_same_shape(self):
        img = make_white_bgr(400, 300)
        out = deskew(img, method="hough", max_angle=15.0)
        assert out.shape == img.shape

    def test_large_angle_skipped(self):
        """Çok büyük açı düzeltme yapılmamalı (orijinal dönmeli)."""
        img = make_white_bgr(400, 300)
        out = deskew(img, method="projection", max_angle=1.0)
        assert out.shape == img.shape

    def test_projection_detect_skew(self):
        binary = make_text_binary()
        angle = detect_skew_projection(binary)
        assert -15 <= angle <= 15


# ─── morphological_clean ───────────────────────

class TestMorphologicalClean:
    def test_opening_no_error(self):
        binary = make_text_binary()
        out = morphological_clean(binary, apply_opening=True, apply_closing=False)
        assert out.shape == binary.shape

    def test_closing_no_error(self):
        binary = make_text_binary()
        out = morphological_clean(binary, apply_opening=False, apply_closing=True)
        assert out.shape == binary.shape

    def test_no_op_returns_same(self):
        binary = make_text_binary()
        out = morphological_clean(binary, apply_opening=False, apply_closing=False)
        np.testing.assert_array_equal(out, binary)


# ─── preprocess_pipeline ───────────────────────

class TestPreprocessPipeline:
    def _minimal_cfg(self):
        return {
            "resize": {"enabled": False},
            "denoising": {"method": "gaussian", "gaussian_kernel": 3},
            "contrast": {"method": "clahe", "clahe_clip_limit": 2.0, "clahe_tile_grid": [8, 8]},
            "thresholding": {"method": "otsu", "otsu_blur_before": True},
            "perspective": {"enabled": False},
            "deskew": {"enabled": False},
            "morphology": {"apply_opening": False, "apply_closing": False, "kernel_size": [3, 3]},
        }

    def test_returns_tuple(self):
        img = make_white_bgr()
        cfg = self._minimal_cfg()
        result = preprocess_pipeline(img, cfg)
        assert isinstance(result, tuple) and len(result) == 2

    def test_binary_has_correct_values(self):
        img = make_white_bgr()
        cfg = self._minimal_cfg()
        _, binary = preprocess_pipeline(img, cfg)
        unique = np.unique(binary)
        assert set(unique).issubset({0, 255})
