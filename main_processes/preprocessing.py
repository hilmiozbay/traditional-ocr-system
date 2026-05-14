"""
preprocessing.py
================
Belge görüntüleri için klasik ön işleme boru hattı.

Adımlar:
  1. Yeniden boyutlandırma (isteğe bağlı)
  2. Gürültü giderme  (Gaussian / Medyan / Bilateral)
  3. Kontrast iyileştirme (CLAHE / Histogram eşitleme)
  4. Eşikleme / Binarizasyon (Otsu / Adaptif)
  5. Eğim tespiti ve düzeltme (Hough / Projeksiyon)
  6. Perspektif düzeltme (warp transform)
  7. Morfolojik temizleme (isteğe bağlı)
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Yardımcı: Görüntü yükleme / kaydetme
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Diskten görüntü yükler; Türkçe / Unicode yolu destekler.

    Args:
        path: Görüntü dosyasının tam yolu.

    Returns:
        BGR formatında NumPy dizisi.

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
        ValueError: Görüntü okunamazsa.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")

    # cv2.imread Türkçe karakterli yolları bazen okuyamaz;
    # np.fromfile ile güvenli yükleme yapıyoruz.
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    logger.debug("Görüntü yüklendi: %s  Boyut: %s", path, img.shape)
    return img


def save_image(img: np.ndarray, path: str) -> None:
    """
    Görüntüyü diske kaydeder (Türkçe yol desteği dahil).

    Args:
        img: Kaydedilecek BGR görüntüsü.
        path: Hedef dosya yolu.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise IOError(f"Görüntü kodlanamadı: {path}")
    buf.tofile(str(p))
    logger.debug("Görüntü kaydedildi: %s", path)


# ─────────────────────────────────────────────
# 2. Yeniden boyutlandırma
# ─────────────────────────────────────────────

def resize_image(
    img: np.ndarray,
    max_width: int = 3000,
    max_height: int = 4000,
    dpi_target: int = 300,
) -> np.ndarray:
    """
    Görüntüyü en-boy oranını koruyarak maksimum boyuta göre ölçekler.

    Args:
        img: Giriş görüntüsü (BGR).
        max_width: İzin verilen maksimum piksel genişliği.
        max_height: İzin verilen maksimum piksel yüksekliği.
        dpi_target: Hedef DPI (bilgi amaçlı loglanır).

    Returns:
        Yeniden boyutlandırılmış görüntü.
    """
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)  # Küçültme yok ise 1.0
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info("Boyut küçültüldü: %dx%d → %dx%d (ölçek=%.2f)", w, h, new_w, new_h, scale)
    else:
        logger.debug("Boyutlandırma gerekmiyor: %dx%d, DPI hedef=%d", w, h, dpi_target)
    return img


# ─────────────────────────────────────────────
# 3. Gürültü giderme
# ─────────────────────────────────────────────

def denoise(
    img: np.ndarray,
    method: str = "gaussian",
    gaussian_kernel: int = 3,
    median_kernel: int = 3,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75,
    bilateral_sigma_space: float = 75,
) -> np.ndarray:
    """
    Çeşitli filtrelerle gürültü giderir.

    Args:
        img: Giriş görüntüsü (BGR veya gri).
        method: 'gaussian' | 'median' | 'bilateral'
        gaussian_kernel: Gaussian çekirdek boyutu (tek sayı).
        median_kernel: Medyan çekirdek boyutu (tek sayı).
        bilateral_d: Bilateral filtre komşuluk çapı.
        bilateral_sigma_color: Renk uzayı sigma değeri.
        bilateral_sigma_space: Koordinat uzayı sigma değeri.

    Returns:
        Gürültüsü giderilmiş görüntü.
    """
    method = method.lower()
    if method == "gaussian":
        k = gaussian_kernel if gaussian_kernel % 2 == 1 else gaussian_kernel + 1
        result = cv2.GaussianBlur(img, (k, k), 0)
    elif method == "median":
        k = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        result = cv2.medianBlur(img, k)
    elif method == "bilateral":
        result = cv2.bilateralFilter(
            img, bilateral_d, bilateral_sigma_color, bilateral_sigma_space
        )
    else:
        logger.warning("Bilinmeyen gürültü giderme yöntemi '%s'; atlanıyor.", method)
        result = img

    logger.debug("Gürültü giderme tamamlandı: yöntem=%s", method)
    return result


# ─────────────────────────────────────────────
# 4. Kontrast iyileştirme
# ─────────────────────────────────────────────

def enhance_contrast(
    img: np.ndarray,
    method: str = "clahe",
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    CLAHE veya global histogram eşitleme ile kontrast artırır.

    Args:
        img: Giriş görüntüsü (BGR veya gri).
        method: 'clahe' | 'histogram_eq' | 'none'
        clahe_clip_limit: CLAHE klip limiti.
        clahe_tile_grid: CLAHE döşeme ızgarası boyutu.

    Returns:
        Kontrast iyileştirilmiş görüntü (gri veya BGR).
    """
    if method == "none":
        return img

    # Griye dönüştür (eğer renkli ise)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if method == "clahe":
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=tuple(clahe_tile_grid),
        )
        result = clahe.apply(gray)
    elif method == "histogram_eq":
        result = cv2.equalizeHist(gray)
    else:
        logger.warning("Bilinmeyen kontrast yöntemi '%s'; atlanıyor.", method)
        result = gray

    logger.debug("Kontrast iyileştirme tamamlandı: yöntem=%s", method)
    return result


# ─────────────────────────────────────────────
# 5. Eşikleme (Binarizasyon)
# ─────────────────────────────────────────────

def binarize(
    img: np.ndarray,
    method: str = "otsu",
    adaptive_block_size: int = 35,
    adaptive_c: int = 10,
    otsu_blur_before: bool = True,
) -> np.ndarray:
    """
    Görüntüyü siyah-beyaz (binary) formata dönüştürür.

    Args:
        img: Gri veya BGR giriş görüntüsü.
        method: 'otsu' | 'adaptive_mean' | 'adaptive_gaussian'
        adaptive_block_size: Adaptif eşikleme blok boyutu (tek sayı, ≥3).
        adaptive_c: Ortalamadan çıkarılacak sabit.
        otsu_blur_before: Otsu öncesi Gaussian blur uygulansın mı?

    Returns:
        Binary görüntü (0 ve 255 piksellerden oluşur).
    """
    # Gri seviyeye indir
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    method = method.lower()
    if method == "otsu":
        if otsu_blur_before:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "adaptive_mean":
        bs = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            bs, adaptive_c,
        )
    elif method == "adaptive_gaussian":
        bs = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs, adaptive_c,
        )
    else:
        logger.warning("Bilinmeyen eşikleme yöntemi '%s'; Otsu kullanılıyor.", method)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    logger.debug("Binarizasyon tamamlandı: yöntem=%s", method)
    return binary


# ─────────────────────────────────────────────
# 6. Eğim tespiti ve düzeltme (Deskew)
# ─────────────────────────────────────────────

def detect_skew_hough(binary_img: np.ndarray, hough_threshold: int = 100) -> float:
    """
    Hough dönüşümü ile sayfanın eğim açısını tahmin eder.

    Args:
        binary_img: İkili (binary) görüntü.
        hough_threshold: Hough dönüşümü için oy eşiği.

    Returns:
        Tespit edilen açı (derece). Pozitif = saat yönü.
    """
    edges = cv2.Canny(binary_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)
    if lines is None:
        logger.debug("Hough: çizgi bulunamadı, açı=0")
        return 0.0

    angles = []
    for line in lines:
        rho, theta = line[0]
        # Yatay çizgiler: theta ≈ 0 veya π (radyan)
        angle_deg = math.degrees(theta)
        if angle_deg < 45:
            angles.append(angle_deg)
        elif angle_deg > 135:
            angles.append(angle_deg - 180)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    logger.debug("Hough eğim açısı: %.2f°", median_angle)
    return median_angle


def detect_skew_projection(binary_img: np.ndarray) -> float:
    """
    Projeksiyon profili ile eğim açısını bulur.
    Binary görüntüyü çeşitli açılarda döndürür; en az varyansa sahip açıyı seçer.

    Args:
        binary_img: İkili (binary) görüntü.

    Returns:
        En iyi eğim açısı (derece).
    """
    best_angle = 0.0
    best_score = -1.0
    h, w = binary_img.shape[:2]

    for angle in np.arange(-15, 15.5, 0.5):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderValue=255)
        # Satır projeksiyon: her satırın beyaz piksel sayısını hesapla
        proj = np.sum(rotated == 0, axis=1).astype(np.float32)
        score = float(np.var(proj))
        if score > best_score:
            best_score = score
            best_angle = angle

    logger.debug("Projeksiyon eğim açısı: %.2f°", best_angle)
    return best_angle


def deskew(
    img: np.ndarray,
    method: str = "hough",
    max_angle: float = 15.0,
    hough_threshold: int = 100,
) -> np.ndarray:
    """
    Görüntünün eğimini tespit edip düzeltir.

    Args:
        img: BGR veya gri görüntü.
        method: 'hough' | 'projection'
        max_angle: Bu açıdan büyük düzeltme yapılmaz (güvenlik).
        hough_threshold: Hough dönüşümü eşiği.

    Returns:
        Eğim düzeltilmiş görüntü.
    """
    # Binary görüntü hazırla
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if method == "hough":
        angle = detect_skew_hough(binary, hough_threshold)
    else:
        angle = detect_skew_projection(binary)

    # Çok büyük açılar güvenilir değil; atla
    if abs(angle) > max_angle:
        logger.warning("Eğim açısı çok büyük (%.2f°); düzeltme atlandı.", angle)
        return img

    if abs(angle) < 0.1:
        logger.debug("Eğim ihmal edilebilir (%.2f°); düzeltme yapılmadı.", angle)
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.info("Eğim düzeltildi: %.2f°", angle)
    return rotated


# ─────────────────────────────────────────────
# 7. Perspektif düzeltme
# ─────────────────────────────────────────────

def detect_document_corners(
    img: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Belgede dört köşe noktasını tespit eder (en büyük dörtgen kontur).

    Args:
        img: BGR görüntü.

    Returns:
        4x1x2 köşe noktaları dizisi veya None (tespit edilemezse).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            logger.debug("Belge köşeleri bulundu.")
            return approx

    logger.debug("Belge köşeleri bulunamadı.")
    return None


def four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Dört nokta perspektif dönüşümü uygular (bird's-eye view).

    Args:
        img: BGR giriş görüntüsü.
        pts: Dörtgenin 4 köşe noktası (4x1x2 veya 4x2 dizisi).

    Returns:
        Perspektif düzeltilmiş görüntü.
    """
    pts = pts.reshape(4, 2).astype(np.float32)

    # Köşeleri sırala: sol-üst, sağ-üst, sağ-alt, sol-alt
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]   # sol-üst
    rect[2] = pts[np.argmax(s)]   # sağ-alt
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # sağ-üst
    rect[3] = pts[np.argmax(diff)]  # sol-alt

    (tl, tr, br, bl) = rect
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    max_w = max(int(w1), int(w2))
    max_h = max(int(h1), int(h2))

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_w, max_h))
    logger.info("Perspektif düzeltme uygulandı: çıktı boyutu %dx%d", max_w, max_h)
    return warped


# ─────────────────────────────────────────────
# 8. Morfolojik işlemler
# ─────────────────────────────────────────────

def morphological_clean(
    binary_img: np.ndarray,
    apply_opening: bool = False,
    apply_closing: bool = False,
    kernel_size: Tuple[int, int] = (3, 3),
) -> np.ndarray:
    """
    Morfolojik opening/closing ile binary görüntüyü temizler.

    Args:
        binary_img: Binary (0/255) giriş görüntüsü.
        apply_opening: Küçük gürültü noktalarını sil (erosion → dilation).
        apply_closing: Küçük boşlukları kapat (dilation → erosion).
        kernel_size: Morfolojik çekirdek boyutu.

    Returns:
        Temizlenmiş binary görüntü.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    result = binary_img.copy()
    if apply_opening:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    if apply_closing:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


# ─────────────────────────────────────────────
# 9. Tam ön işleme boru hattı
# ─────────────────────────────────────────────

def preprocess_pipeline(img: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Konfigürasyona göre tam ön işleme boru hattını çalıştırır.

    Args:
        img: Ham BGR görüntü.
        cfg: config.yaml'dan yüklenen 'preprocessing' bloğu.

    Returns:
        (işlenmiş_bgr, binary) demeti:
          - işlenmiş_bgr: Eğim/perspektif düzeltilmiş renkli görüntü
          - binary     : OCR'a hazır siyah-beyaz görüntü
    """
    # 1. Boyutlandırma
    if cfg.get("resize", {}).get("enabled", True):
        r = cfg["resize"]
        img = resize_image(img, r["max_width"], r["max_height"], r["dpi_target"])

    # 2. Gürültü giderme
    d = cfg.get("denoising", {})
    img = denoise(img, method=d.get("method", "gaussian"),
                  gaussian_kernel=d.get("gaussian_kernel", 3),
                  median_kernel=d.get("median_kernel", 3))

    # 3. Kontrast iyileştirme (gri çıktı üretir)
    c = cfg.get("contrast", {})
    gray_enhanced = enhance_contrast(img, method=c.get("method", "clahe"),
                                     clahe_clip_limit=c.get("clahe_clip_limit", 2.0),
                                     clahe_tile_grid=tuple(c.get("clahe_tile_grid", [8, 8])))

    # 4. Eşikleme
    t = cfg.get("thresholding", {})
    binary = binarize(gray_enhanced, method=t.get("method", "otsu"),
                      adaptive_block_size=t.get("adaptive_block_size", 35),
                      adaptive_c=t.get("adaptive_c", 10),
                      otsu_blur_before=t.get("otsu_blur_before", True))

    # 5. Perspektif düzeltme (BGR üzerinde yapıyoruz)
    p_cfg = cfg.get("perspective", {})
    if p_cfg.get("enabled", True) and p_cfg.get("auto_detect", True):
        corners = detect_document_corners(img)
        if corners is not None:
            img = four_point_transform(img, corners)
            # Binary'yi de yeniden hesapla
            gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Eğim düzeltme
    sk = cfg.get("deskew", {})
    if sk.get("enabled", True):
        img = deskew(img, method=sk.get("method", "hough"),
                     max_angle=sk.get("max_angle", 15.0),
                     hough_threshold=sk.get("hough_threshold", 100))
        # Binary'yi yeniden üret
        gray3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 7. Morfolojik temizleme
    m = cfg.get("morphology", {})
    binary = morphological_clean(
        binary,
        apply_opening=m.get("apply_opening", False),
        apply_closing=m.get("apply_closing", False),
        kernel_size=tuple(m.get("kernel_size", [3, 3])),
    )

    logger.info("Ön işleme boru hattı tamamlandı.")
    return img, binary
