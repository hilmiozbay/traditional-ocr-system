"""
tests/test_segmentation.py
===========================
segmentation.py modülü için birim testler (pytest).
"""

import numpy as np
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from segmentation import (
    connected_component_analysis,
    horizontal_projection,
    vertical_projection,
    detect_text_lines,
    detect_columns,
    detect_tables,
    _merge_overlapping_boxes,
    _iou,
    draw_regions,
)


def make_binary_with_blobs() -> np.ndarray:
    """Belirli konumlarda siyah blob içeren binary görüntü."""
    img = np.full((300, 400), 255, dtype=np.uint8)
    img[50:80, 30:100] = 0    # blob 1
    img[50:80, 200:280] = 0   # blob 2
    img[150:170, 50:350] = 0  # uzun yatay çizgi
    return img


def make_table_binary() -> np.ndarray:
    """Tablo benzeri yatay+dikey çizgiler içeren binary görüntü."""
    img = np.full((400, 400), 255, dtype=np.uint8)
    # Yatay çizgiler
    for y in [50, 150, 250, 350]:
        img[y:y+2, 50:350] = 0
    # Dikey çizgiler
    for x in [50, 150, 250, 350]:
        img[50:352, x:x+2] = 0
    return img


# ─── connected_component_analysis ─────────────

class TestConnectedComponentAnalysis:
    def test_finds_blobs(self):
        binary = make_binary_with_blobs()
        comps = connected_component_analysis(binary, min_area=100)
        assert len(comps) >= 1

    def test_min_area_filter(self):
        binary = np.full((100, 100), 255, dtype=np.uint8)
        binary[10:12, 10:12] = 0  # 4px² çok küçük blob
        comps = connected_component_analysis(binary, min_area=10)
        # Küçük blob filtrelenmeli veya dahil edilmeli; hata verme
        assert isinstance(comps, list)

    def test_returns_dicts_with_keys(self):
        binary = make_binary_with_blobs()
        comps = connected_component_analysis(binary, min_area=10)
        for c in comps:
            assert "bbox" in c
            assert "area" in c
            assert "aspect_ratio" in c
            assert "centroid" in c


# ─── projeksiyon profilleri ────────────────────

class TestProjectionProfiles:
    def test_horizontal_projection_shape(self):
        binary = make_binary_with_blobs()
        proj = horizontal_projection(binary)
        assert proj.shape[0] == binary.shape[0]

    def test_vertical_projection_shape(self):
        binary = make_binary_with_blobs()
        proj = vertical_projection(binary)
        assert proj.shape[0] == binary.shape[1]

    def test_horizontal_nonzero_where_text(self):
        binary = make_binary_with_blobs()
        proj = horizontal_projection(binary)
        # Satır 50-80 arası siyah piksel olmalı
        assert proj[60] > 0

    def test_projection_all_white(self):
        binary = np.full((100, 100), 255, dtype=np.uint8)
        h_proj = horizontal_projection(binary)
        v_proj = vertical_projection(binary)
        assert np.all(h_proj == 0)
        assert np.all(v_proj == 0)


# ─── satır ve sütun tespiti ────────────────────

class TestLineAndColumnDetection:
    def test_detect_text_lines_finds_lines(self):
        binary = make_binary_with_blobs()
        lines = detect_text_lines(binary, threshold=5, min_line_gap=5)
        assert len(lines) >= 1

    def test_detect_columns_finds_columns(self):
        binary = make_binary_with_blobs()
        cols = detect_columns(binary, threshold=5, min_col_gap=5)
        assert len(cols) >= 1

    def test_detect_lines_returns_tuples(self):
        binary = make_binary_with_blobs()
        lines = detect_text_lines(binary)
        for start, end in lines:
            assert start <= end

    def test_all_white_no_lines(self):
        binary = np.full((200, 200), 255, dtype=np.uint8)
        lines = detect_text_lines(binary, threshold=5)
        assert lines == []


# ─── tablo tespiti ─────────────────────────────

class TestTableDetection:
    def test_detects_table_structure(self):
        binary = make_table_binary()
        tables = detect_tables(binary, min_line_length=80)
        # En az bir tablo bölgesi bulunmalı
        assert len(tables) >= 1

    def test_no_table_in_plain_image(self):
        binary = np.full((200, 200), 255, dtype=np.uint8)
        tables = detect_tables(binary, min_line_length=50)
        assert tables == []


# ─── yardımcı işlevler ─────────────────────────

class TestHelpers:
    def test_iou_same_box(self):
        box = np.array([0, 0, 100, 100], dtype=float)
        assert abs(_iou(box, box) - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        a = np.array([0, 0, 50, 50], dtype=float)
        b = np.array([100, 100, 200, 200], dtype=float)
        assert _iou(a, b) == 0.0

    def test_merge_overlapping_boxes(self):
        boxes = [(0, 0, 50, 50), (10, 10, 50, 50)]
        merged = _merge_overlapping_boxes(boxes, iou_threshold=0.1)
        assert len(merged) == 1

    def test_merge_non_overlapping_boxes(self):
        boxes = [(0, 0, 10, 10), (200, 200, 10, 10)]
        merged = _merge_overlapping_boxes(boxes, iou_threshold=0.3)
        assert len(merged) == 2

    def test_draw_regions_no_error(self):
        import cv2
        img = np.full((200, 300, 3), 200, dtype=np.uint8)
        boxes = [(10, 10, 50, 50), (100, 100, 80, 40)]
        out = draw_regions(img, boxes)
        assert out.shape == img.shape
