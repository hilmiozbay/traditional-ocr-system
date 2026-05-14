"""
segmentation.py
===============
Belge analizi ve bölütleme (segmentation) modülü.

İşlevler:
  - Bağlı bileşen analizi (connected components)
  - MSER ile metin bölgesi tespiti
  - Projeksiyon profili ile satır/sütun tespiti
  - Morfologik yöntemle tablo tespiti
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Bounding box tip tanımı: (x, y, w, h)
BBox = Tuple[int, int, int, int]


# ─────────────────────────────────────────────
# 1. Bağlı bileşen analizi
# ─────────────────────────────────────────────

def connected_component_analysis(
    binary_img: np.ndarray,
    min_area: int = 50,
    max_area: int = 500_000,
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 10.0,
) -> List[Dict]:
    """
    Binary görüntüdeki bağlı bileşenleri analiz eder ve filtreler.

    Args:
        binary_img: 0/255 değerli binary görüntü (metin=siyah, zemin=beyaz).
        min_area: Minimum bileşen alanı (piksel²).
        max_area: Maksimum bileşen alanı (piksel²).
        min_aspect_ratio: Minimum en-boy oranı (w/h).
        max_aspect_ratio: Maksimum en-boy oranı (w/h).

    Returns:
        Her bileşen için {'bbox', 'area', 'aspect_ratio', 'centroid'} içeren liste.
    """
    # Metni beyaz, zemini siyah yap (bağlı bileşen analizi için)
    inv = cv2.bitwise_not(binary_img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inv, connectivity=8
    )

    components = []
    for i in range(1, num_labels):  # 0 = arka plan
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        if area < min_area or area > max_area:
            continue
        aspect = w / h if h > 0 else 0
        if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
            continue

        components.append({
            "bbox": (x, y, w, h),
            "area": area,
            "aspect_ratio": round(aspect, 3),
            "centroid": (round(cx, 1), round(cy, 1)),
        })

    logger.info("Bağlı bileşen analizi: %d bileşen bulundu (filtreden sonra).", len(components))
    return components


# ─────────────────────────────────────────────
# 2. MSER ile metin bölgesi tespiti
# ─────────────────────────────────────────────

def detect_text_regions_mser(
    img: np.ndarray,
    delta: int = 5,
    min_area: int = 60,
    max_area: int = 14_400,
) -> List[BBox]:
    """
    MSER (Maximally Stable Extremal Regions) ile metin bölgelerini tespit eder.

    Args:
        img: BGR veya gri giriş görüntüsü.
        delta: MSER delta parametresi (kararlılık eşiği).
        min_area: Minimum bölge alanı.
        max_area: Maksimum bölge alanı.

    Returns:
        (x, y, w, h) bounding box listesi.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # OpenCV 4.x: keyword argümanları kaldırıldı, pozisyonel kullan
    mser = cv2.MSER_create(delta, min_area, max_area)
    regions, bboxes = mser.detectRegions(gray)
    boxes: List[BBox] = [(int(x), int(y), int(w), int(h)) for x, y, w, h in bboxes]

    # NMS benzeri birleştirme: iç içe geçen kutuları birleştir
    boxes = _merge_overlapping_boxes(boxes, iou_threshold=0.3)

    logger.info("MSER tespiti: %d metin bölgesi.", len(boxes))
    return boxes


def _merge_overlapping_boxes(
    boxes: List[BBox],
    iou_threshold: float = 0.3,
) -> List[BBox]:
    """
    Çakışan bounding box'ları birleştirir (greedy merge).

    Args:
        boxes: (x, y, w, h) listesi.
        iou_threshold: Bu IoU değerinin üzerindeki kutular birleştirilir.

    Returns:
        Birleştirilmiş kutular listesi.
    """
    if not boxes:
        return []

    # (x1, y1, x2, y2) formatına çevir
    rects = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=np.float32)
    keep = []
    used = [False] * len(rects)

    for i in range(len(rects)):
        if used[i]:
            continue
        group = [rects[i]]
        for j in range(i + 1, len(rects)):
            if used[j]:
                continue
            if _iou(rects[i], rects[j]) > iou_threshold:
                group.append(rects[j])
                used[j] = True
        merged = np.array(group)
        x1 = int(merged[:, 0].min())
        y1 = int(merged[:, 1].min())
        x2 = int(merged[:, 2].max())
        y2 = int(merged[:, 3].max())
        keep.append((x1, y1, x2 - x1, y2 - y1))
        used[i] = True

    return keep


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """İki bounding box arasındaki Intersection over Union değeri."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# 3. Projeksiyon profili ile satır/sütun tespiti
# ─────────────────────────────────────────────

def horizontal_projection(binary_img: np.ndarray) -> np.ndarray:
    """
    Yatay projeksiyon profili: her satırdaki siyah piksel sayısı.

    Args:
        binary_img: 0/255 binary görüntü (metin=siyah).

    Returns:
        1-D NumPy dizisi, her eleman bir satırın siyah piksel sayısı.
    """
    return np.sum(binary_img == 0, axis=1).astype(np.int32)


def vertical_projection(binary_img: np.ndarray) -> np.ndarray:
    """
    Dikey projeksiyon profili: her sütundaki siyah piksel sayısı.

    Args:
        binary_img: 0/255 binary görüntü (metin=siyah).

    Returns:
        1-D NumPy dizisi, her eleman bir sütunun siyah piksel sayısı.
    """
    return np.sum(binary_img == 0, axis=0).astype(np.int32)


def detect_text_lines(
    binary_img: np.ndarray,
    threshold: int = 5,
    min_line_gap: int = 10,
) -> List[Tuple[int, int]]:
    """
    Yatay projeksiyon ile metin satırlarının y aralıklarını bulur.

    Args:
        binary_img: Binary görüntü.
        threshold: Satır sayılması için minimum piksel sayısı.
        min_line_gap: Satırlar arası minimum boşluk (piksel).

    Returns:
        (y_start, y_end) satır aralıkları listesi.
    """
    profile = horizontal_projection(binary_img)
    in_line = False
    lines: List[Tuple[int, int]] = []
    y_start = 0

    for y, val in enumerate(profile):
        if not in_line and val >= threshold:
            in_line = True
            y_start = y
        elif in_line and val < threshold:
            in_line = False
            if (y - y_start) >= min_line_gap:
                lines.append((y_start, y))

    if in_line:
        lines.append((y_start, len(profile) - 1))

    logger.info("Satır tespiti: %d satır bulundu.", len(lines))
    return lines


def detect_columns(
    binary_img: np.ndarray,
    threshold: int = 5,
    min_col_gap: int = 10,
) -> List[Tuple[int, int]]:
    """
    Dikey projeksiyon ile sütun x aralıklarını bulur.

    Args:
        binary_img: Binary görüntü.
        threshold: Sütun sayılması için minimum piksel sayısı.
        min_col_gap: Sütunlar arası minimum boşluk (piksel).

    Returns:
        (x_start, x_end) sütun aralıkları listesi.
    """
    profile = vertical_projection(binary_img)
    in_col = False
    cols: List[Tuple[int, int]] = []
    x_start = 0

    for x, val in enumerate(profile):
        if not in_col and val >= threshold:
            in_col = True
            x_start = x
        elif in_col and val < threshold:
            in_col = False
            if (x - x_start) >= min_col_gap:
                cols.append((x_start, x))

    if in_col:
        cols.append((x_start, len(profile) - 1))

    logger.info("Sütun tespiti: %d sütun bulundu.", len(cols))
    return cols


# ─────────────────────────────────────────────
# 4. Tablo tespiti (morfolojik yöntem)
# ─────────────────────────────────────────────

def detect_tables(
    binary_img: np.ndarray,
    min_line_length: int = 100,
    kernel_horizontal: Tuple[int, int] = (1, 40),
    kernel_vertical: Tuple[int, int] = (40, 1),
) -> List[BBox]:
    """
    Yatay ve dikey çizgileri birleştirerek tablo bölgelerini tespit eder.

    Yöntem:
      1. Yatay morfolojik çizgileri çıkar.
      2. Dikey morfolojik çizgileri çıkar.
      3. İkisini birleştir → tablo ızgarası.
      4. Konturları bul → tablo bounding box'ları.

    Args:
        binary_img: Binary (0/255) görüntü.
        min_line_length: Minimum çizgi uzunluğu (piksel).
        kernel_horizontal: Yatay çizgi için morfolojik çekirdek boyutu.
        kernel_vertical: Dikey çizgi için morfolojik çekirdek boyutu.

    Returns:
        (x, y, w, h) tablo kutusu listesi.
    """
    inv = cv2.bitwise_not(binary_img)

    # Yatay çizgiler
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_horizontal)
    horizontal_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh)

    # Dikey çizgiler
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_vertical)
    vertical_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kv)

    # Birleştir
    table_mask = cv2.add(horizontal_lines, vertical_lines)

    # Konturları bul
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables: List[BBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_line_length and h >= min_line_length:
            tables.append((x, y, w, h))

    logger.info("Tablo tespiti: %d tablo bölgesi bulundu.", len(tables))
    return tables


# ─────────────────────────────────────────────
# 5. Görsel hata ayıklama yardımcısı
# ─────────────────────────────────────────────

def draw_regions(
    img: np.ndarray,
    boxes: List[BBox],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str = "",
) -> np.ndarray:
    """
    Bounding box'ları görüntü üzerine çizer (hata ayıklama için).

    Args:
        img: BGR giriş görüntüsü.
        boxes: (x, y, w, h) kutu listesi.
        color: BGR renk.
        thickness: Çizgi kalınlığı.
        label: Kutu üzerine yazılacak etiket öneki.

    Returns:
        Kutular çizilmiş görüntü kopyası.
    """
    out = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        if label:
            cv2.putText(out, f"{label}{i}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


# ─────────────────────────────────────────────
# 6. Tam segmentasyon boru hattı
# ─────────────────────────────────────────────

def segment_document(
    img: np.ndarray,
    binary_img: np.ndarray,
    cfg: dict,
) -> Dict:
    """
    Belge üzerinde tam segmentasyon analizini çalıştırır.

    Args:
        img: BGR giriş görüntüsü.
        binary_img: Ön işlemden geçmiş binary görüntü.
        cfg: config.yaml'dan 'segmentation' bloğu.

    Returns:
        {
          'text_regions': [...],  # Metin bounding box'ları
          'text_lines': [...],    # Satır aralıkları
          'columns': [...],       # Sütun aralıkları
          'tables': [...],        # Tablo bounding box'ları
          'components': [...],    # Tüm bağlı bileşenler
        }
    """
    # Bağlı bileşenler
    cc_cfg = cfg.get("connected_components", {})
    components = connected_component_analysis(
        binary_img,
        min_area=cc_cfg.get("min_area", 50),
        max_area=cc_cfg.get("max_area", 500_000),
        min_aspect_ratio=cc_cfg.get("min_aspect_ratio", 0.1),
        max_aspect_ratio=cc_cfg.get("max_aspect_ratio", 10.0),
    )

    # Metin bölgeleri
    td_cfg = cfg.get("text_detection", {})
    method = td_cfg.get("method", "mser")
    if method == "mser":
        text_regions = detect_text_regions_mser(
            img,
            delta=td_cfg.get("mser_delta", 5),
            min_area=td_cfg.get("mser_min_area", 60),
            max_area=td_cfg.get("mser_max_area", 14400),
        )
    else:
        # Yedek: bileşen kutuları kullan
        text_regions = [c["bbox"] for c in components]

    # Satır ve sütun tespiti
    pr_cfg = cfg.get("projection", {})
    text_lines = detect_text_lines(
        binary_img,
        threshold=pr_cfg.get("horizontal_threshold", 5),
        min_line_gap=pr_cfg.get("min_line_gap", 10),
    )
    columns = detect_columns(
        binary_img,
        threshold=pr_cfg.get("vertical_threshold", 5),
        min_col_gap=pr_cfg.get("min_line_gap", 10),
    )

    # Tablo tespiti
    result_tables: List[BBox] = []
    tb_cfg = cfg.get("table_detection", {})
    if tb_cfg.get("enabled", True):
        result_tables = detect_tables(
            binary_img,
            min_line_length=tb_cfg.get("min_line_length", 100),
            kernel_horizontal=tuple(tb_cfg.get("kernel_horizontal", [1, 40])),
            kernel_vertical=tuple(tb_cfg.get("kernel_vertical", [40, 1])),
        )

    return {
        "text_regions": text_regions,
        "text_lines": text_lines,
        "columns": columns,
        "tables": result_tables,
        "components": components,
    }
