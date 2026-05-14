"""
ocr.py
======
Tesseract OCR motoru ile metin tanıma modülü.

İşlevler:
  - Tam sayfa OCR
  - Bölge bazlı OCR (crop + tanıma)
  - Karakter / kelime güven skoru analizi
  - Çok dilli ve Türkçe destek
  - Hough-tabanlı görüntü ön hazırlık (OCR'a özgü)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import Output
except ImportError as exc:
    raise ImportError(
        "pytesseract yüklü değil. Lütfen: pip install pytesseract"
    ) from exc

# ── Windows'ta Tesseract otomatik yol tespiti ─────────────────
import os as _os, shutil as _shutil

def _find_tesseract() -> str:
    """
    Tesseract yürütülebilir dosyasını bulur; önce PATH, sonra
    Windows standart kurulum konumlarını dener.
    """
    # PATH'te var mı?
    path_hit = _shutil.which("tesseract")
    if path_hit:
        return path_hit
    # Standart kurulum konumları
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        _os.path.join(_os.environ.get("LOCALAPPDATA", ""), "Tesseract-OCR", "tesseract.exe"),
    ]
    for c in candidates:
        if _os.path.isfile(c):
            return c
    return "tesseract"  # Son çare; hata varsa kullanıcıya gösterilir

_tess_cmd = _find_tesseract()
pytesseract.pytesseract.tesseract_cmd = _tess_cmd

logger = logging.getLogger(__name__)
logger.debug("Tesseract yolu: %s", _tess_cmd)


# ─────────────────────────────────────────────
# Veri yapıları
# ─────────────────────────────────────────────

@dataclass
class WordResult:
    """Tek bir tanınan kelimenin tüm bilgileri."""
    text: str
    confidence: float          # 0-100
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    line_num: int
    block_num: int
    page_num: int


@dataclass
class OCRResult:
    """Bir görüntü için tam OCR sonucu."""
    full_text: str
    words: List[WordResult] = field(default_factory=list)
    avg_confidence: float = 0.0
    language: str = "tur+eng"
    psm: int = 3
    image_shape: Tuple[int, int] = (0, 0)  # (h, w)


# ─────────────────────────────────────────────
# 1. Tesseract yapılandırma yardımcısı
# ─────────────────────────────────────────────

def build_tesseract_config(
    psm: int = 3,
    oem: int = 1,
    dpi: int = 300,
    extra: str = "",
) -> str:
    """
    Tesseract komut satırı yapılandırma dizesini oluşturur.

    PSM (Page Segmentation Mode) değerleri:
      0  – Sadece yön/betik tespiti
      3  – Tam otomatik segmentasyon (varsayılan)
      4  – Tek sütun, değişken boyutlu metin
      6  – Tek tip metin bloğu
      7  – Tek satır
      8  – Tek kelime
      11 – Seyrek metin
      13 – Ham satır

    OEM (OCR Engine Mode):
      0 – Legacy Tesseract
      1 – LSTM (önerilir)
      3 – Default (otomatik)

    Args:
        psm: Sayfa bölütleme modu.
        oem: Motor modu.
        dpi: Görüntü DPI değeri.
        extra: Ek parametreler (örn. "--tessdata-dir /path").

    Returns:
        Tesseract config dizesi.
    """
    config = f"--psm {psm} --oem {oem} --dpi {dpi}"
    if extra:
        config += f" {extra}"
    return config


# ─────────────────────────────────────────────
# 2. Görüntü ön hazırlık (OCR'a özgü)
# ─────────────────────────────────────────────

def prepare_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Tesseract için son görüntü optimizasyonları.

    Adımlar:
      - Gri seviyeye indir
      - Ölçekleme (OCR için min 300 DPI benzeri boyut önerilir)
      - Kenar keskinleştirme (unsharp mask)
      - Binary'ye çevir (Otsu)

    Args:
        img: BGR veya gri giriş görüntüsü.

    Returns:
        OCR'a hazır görüntü (gri veya binary).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Görüntü çok küçükse 2x büyüt
    h, w = gray.shape
    if w < 1000:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        logger.debug("OCR hazırlık: görüntü 2x büyütüldü → %dx%d", w * 2, h * 2)

    # Unsharp mask ile netliği artır
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # Otsu eşikleme
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ─────────────────────────────────────────────
# 3. Tam sayfa OCR
# ─────────────────────────────────────────────

def ocr_full_page(
    img: np.ndarray,
    language: str = "tur+eng",
    psm: int = 3,
    oem: int = 1,
    dpi: int = 300,
    min_confidence: float = 30.0,
    config_extra: str = "",
) -> OCRResult:
    """
    Tam görüntü üzerinde Tesseract OCR uygular.

    Args:
        img: BGR veya gri görüntü.
        language: Tesseract dil kodu (örn. 'tur', 'eng', 'tur+eng').
        psm: Sayfa bölütleme modu.
        oem: OCR motor modu.
        dpi: Görüntü DPI değeri.
        min_confidence: Bu güven skorunun altındaki kelimeler filtrelenir.
        config_extra: Ek Tesseract parametreleri.

    Returns:
        OCRResult nesnesi.
    """
    config = build_tesseract_config(psm, oem, dpi, config_extra)
    ocr_img = prepare_for_ocr(img)

    try:
        # Kelime detaylı çıktı
        data = pytesseract.image_to_data(
            ocr_img,
            lang=language,
            config=config,
            output_type=Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract bulunamadı. Lütfen https://github.com/UB-Mannheim/tesseract/wiki "
            "adresinden yükleyin ve PATH'e ekleyin."
        ) from exc

    words: List[WordResult] = []
    n = len(data["text"])
    for i in range(n):
        raw_text = data["text"][i].strip()
        if not raw_text:
            continue
        conf = float(data["conf"][i])
        if conf < min_confidence:
            continue
        words.append(WordResult(
            text=raw_text,
            confidence=conf,
            bbox=(
                int(data["left"][i]),
                int(data["top"][i]),
                int(data["width"][i]),
                int(data["height"][i]),
            ),
            line_num=int(data["line_num"][i]),
            block_num=int(data["block_num"][i]),
            page_num=int(data["page_num"][i]),
        ))

    full_text = pytesseract.image_to_string(ocr_img, lang=language, config=config)
    avg_conf = float(np.mean([w.confidence for w in words])) if words else 0.0

    result = OCRResult(
        full_text=full_text.strip(),
        words=words,
        avg_confidence=round(avg_conf, 2),
        language=language,
        psm=psm,
        image_shape=ocr_img.shape[:2],
    )
    logger.info(
        "OCR tamamlandı: %d kelime, ort. güven=%.1f%%, PSM=%d, dil=%s",
        len(words), avg_conf, psm, language,
    )
    return result


# ─────────────────────────────────────────────
# 4. Bölge bazlı OCR
# ─────────────────────────────────────────────

def ocr_region(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    language: str = "tur+eng",
    psm: int = 6,
    oem: int = 1,
    dpi: int = 300,
    padding: int = 5,
) -> str:
    """
    Belirtilen bölgeyi kırpıp OCR uygular.

    Args:
        img: Tam sayfa BGR görüntüsü.
        bbox: (x, y, w, h) bölge koordinatları.
        language: Tesseract dil kodu.
        psm: Sayfa bölütleme modu (bölge için PSM=6 önerilir).
        oem: OCR motor modu.
        dpi: Görüntü DPI değeri.
        padding: Bölgeye eklenen kenar boşluğu (piksel).

    Returns:
        Tanınan metin dizesi.
    """
    x, y, w, h = bbox
    H, W = img.shape[:2]

    # Sınırları aşma kontrolü
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        logger.warning("Boş kırpma bölgesi: bbox=%s", bbox)
        return ""

    config = build_tesseract_config(psm, oem, dpi)
    ocr_crop = prepare_for_ocr(crop)

    try:
        text = pytesseract.image_to_string(ocr_crop, lang=language, config=config)
    except Exception as exc:
        logger.error("Bölge OCR hatası: %s", exc)
        return ""

    return text.strip()


# ─────────────────────────────────────────────
# 5. Çoklu bölge OCR (satır/tablo hücreleri)
# ─────────────────────────────────────────────

def ocr_regions_batch(
    img: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    language: str = "tur+eng",
    psm: int = 7,
    oem: int = 1,
    dpi: int = 300,
) -> List[Dict]:
    """
    Birden fazla bölge üzerinde sırayla OCR uygular.

    Args:
        img: Tam sayfa BGR görüntüsü.
        bboxes: (x, y, w, h) kutu listesi.
        language: Tesseract dil kodu.
        psm: Sayfa bölütleme modu (PSM=7 tek satır için iyi çalışır).
        oem: OCR motor modu.
        dpi: Görüntü DPI değeri.

    Returns:
        [{'bbox': ..., 'text': ..., 'confidence': ...}, ...] listesi.
    """
    results = []
    for bbox in bboxes:
        text = ocr_region(img, bbox, language=language, psm=psm, oem=oem, dpi=dpi)
        results.append({"bbox": bbox, "text": text, "confidence": None})
        logger.debug("Bölge OCR: bbox=%s → '%s'", bbox, text[:40])
    return results


# ─────────────────────────────────────────────
# 6. Güven skoru analizi
# ─────────────────────────────────────────────

def analyze_confidence(result: OCRResult) -> Dict:
    """
    OCR sonucunun güven skoru istatistiklerini hesaplar.

    Args:
        result: ocr_full_page() çıktısı.

    Returns:
        {
          'avg': float,
          'min': float,
          'max': float,
          'low_conf_words': [WordResult, ...]  # < 50 güven
          'word_count': int,
        }
    """
    if not result.words:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "low_conf_words": [], "word_count": 0}

    confs = [w.confidence for w in result.words]
    low_conf = [w for w in result.words if w.confidence < 50]

    return {
        "avg": round(float(np.mean(confs)), 2),
        "min": round(float(np.min(confs)), 2),
        "max": round(float(np.max(confs)), 2),
        "low_conf_words": low_conf,
        "word_count": len(result.words),
    }


# ─────────────────────────────────────────────
# 7. OCR boru hattı
# ─────────────────────────────────────────────

def run_ocr_pipeline(
    img: np.ndarray,
    binary_img: np.ndarray,
    cfg: dict,
    text_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> OCRResult:
    """
    Konfigürasyona göre tam OCR boru hattını çalıştırır.

    Args:
        img: BGR giriş görüntüsü (eğim/perspektif düzeltilmiş).
        binary_img: Binarize edilmiş görüntü.
        cfg: config.yaml'dan 'ocr' bloğu.
        text_regions: Segmentasyondan gelen metin bölgeleri (isteğe bağlı).

    Returns:
        OCRResult nesnesi.
    """
    tess_cfg = cfg.get("tesseract", {})
    conf_cfg = cfg.get("confidence", {})

    result = ocr_full_page(
        img=binary_img,
        language=tess_cfg.get("language", "tur+eng"),
        psm=tess_cfg.get("psm", 3),
        oem=tess_cfg.get("oem", 1),
        dpi=tess_cfg.get("dpi", 300),
        min_confidence=conf_cfg.get("min_word_confidence", 30),
        config_extra=tess_cfg.get("config_extra", ""),
    )

    logger.info(
        "OCR boru hattı tamamlandı | kelime=%d | ort. güven=%.1f%%",
        len(result.words),
        result.avg_confidence,
    )
    return result
