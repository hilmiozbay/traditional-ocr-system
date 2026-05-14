"""
postprocessing.py
=================
OCR sonrası metin temizleme ve normalleştirme modülü.

İşlevler:
  - Unicode ve kontrol karakteri temizleme
  - Türkçe özel OCR hata düzeltmeleri
  - Beyaz boşluk normalleştirme
  - Yapılandırılmış çıktı üretimi (TXT / JSON / CSV)
"""

import csv
import io
import json
import logging
import re
import unicodedata
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from main_processes.ocr import OCRResult, WordResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Türkçe OCR hata tablosu
# ─────────────────────────────────────────────

# Sik karsilasilan OCR karakter karisikliklari (buyuk/kucuk harf duyarli)
# NOT: Siralama onemlidir — daha spesifik kurallar once gelir.
TURKISH_OCR_REPLACEMENTS: List[tuple] = [
    # Turkce ozel harf / ASCII karisikliklari
    # Tesseract bold/italic fontta 'i'/'l'/'|' uretiyor
    (r"(?<![a-z\u011f\xfc\u015f\u0131\xf6\xe7A-Z\u011e\xdc\u015e\u0130\xd6\xc7])l(?=[a-z\u011f\xfc\u015f\u0131\xf6\xe7A-Z\u011e\xdc\u015e\u0130\xd6\xc7])", "i"),
    (r"\bl\b", "i"),
    (r"\|", "i"),
    # Rakam harf karisikliklari
    (r"\b0\b", "O"),
    (r"\b1\b", "i"),
    # 'ii' -> 'il' / 'li' duzeltmesi
    (r"(?<=[aeiou\u0131\xfc\xf6])ii(?=[a-z])", "il"),
    (r"(?<=[a-z])ii(?=[aeiou\u0131\xfc\xf6])", "li"),
    (r"(?<=[a-z])iii(?=[a-z])", "ili"),
    # Turkce karakter ASCII ikameleri
    ("c,", "\xe7"),
    ("s,", "\u015f"),
    ("g`", "\u011f"),
    ("G`", "\u011e"),
    ("u:", "\xfc"),
    ("U:", "\xdc"),
    ("o:", "\xf6"),
    ("O:", "\xd6"),
    # Gozlemlenen spesifik bozulmalar (bu goruntu icin)
    (r"\binceiemek\b", "incelemek"),
    (r"\baractirmak\b", "ara\u015ft\u0131rmak"),
    (r"\be\u015eie\u015ftiriniz\b", "e\u015fle\u015ftiriniz"),
    (r"\bkar\u015f\u0131i\u0131kiar\u0131\b", "kar\u015f\u0131l\u0131klar\u0131"),
    (r"\bbeiiriemek\b", "belirlemek"),
    (r"\boiabiiiriik\b", "olabilirlik"),
    # Noktalama
    (r"\.{2,}", "..."),
    (r"\u2014{2,}", "\u2014"),
]

# Kural bazli duzeltmeler icin regex onceden derle
_COMPILED_TR = [
    (re.compile(pat), repl)
    for pat, repl in TURKISH_OCR_REPLACEMENTS
]



# ─────────────────────────────────────────────
# 2. Metin normalleştirme fonksiyonları
# ─────────────────────────────────────────────

def fix_unicode(text: str) -> str:
    """
    Unicode normalleştirme (NFC) uygular; bozuk karakterleri onarır.

    Args:
        text: Ham OCR metni.

    Returns:
        NFC normalize edilmiş metin.
    """
    return unicodedata.normalize("NFC", text)


def remove_control_chars(text: str) -> str:
    """
    Satır sonu ve sekme dışındaki kontrol karakterlerini kaldırır.

    Args:
        text: Giriş metni.

    Returns:
        Temizlenmiş metin.
    """
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf") or ch in ("\n", "\t", "\r")
    )


def normalize_whitespace(text: str) -> str:
    """
    Fazla boşlukları, sekmeleri ve satır başlarını normalleştirir.

    Args:
        text: Giriş metni.

    Returns:
        Normalleştirilmiş metin.
    """
    # Satır içi çoklu boşluklar → tek boşluk
    text = re.sub(r"[ \t]+", " ", text)
    # Üçten fazla art arda satır sonu → iki satır sonu
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Satır başı/sonu boşlukları temizle
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines)


def fix_punctuation(text: str) -> str:
    """
    Noktalama işaretlerini düzeltir (boşluk kuralları vb.).

    Args:
        text: Giriş metni.

    Returns:
        Düzeltilmiş metin.
    """
    # Noktalama öncesindeki boşlukları kaldır (', . ! ? ; :')
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    # Virgül/nokta sonrası boşluk ekle (eğer yoksa)
    text = re.sub(r"([.,!?;:])(?=[^\s\d\n])", r"\1 ", text)
    return text


def apply_turkish_corrections(text: str) -> str:
    """
    Türkçeye özgü OCR hatalarını düzeltir.

    Args:
        text: Giriş metni.

    Returns:
        Düzeltilmiş metin.
    """
    for pattern, replacement in _COMPILED_TR:
        text = pattern.sub(replacement, text)
    return text


def normalize_text(
    text: str,
    fix_unicode_flag: bool = True,
    remove_control: bool = True,
    normalize_ws: bool = True,
    fix_punct: bool = True,
    turkish_corrections: bool = True,
) -> str:
    """
    Tüm metin normalleştirme adımlarını sırayla uygular.

    Args:
        text: Ham OCR metni.
        fix_unicode_flag: NFC normalleştirme yapılsın mı?
        remove_control: Kontrol karakterleri temizlensin mi?
        normalize_ws: Boşluklar normalleştirilsin mi?
        fix_punct: Noktalama düzeltilsin mi?
        turkish_corrections: Türkçe OCR hata düzeltmeleri yapılsın mı?

    Returns:
        Normalleştirilmiş metin.
    """
    if fix_unicode_flag:
        text = fix_unicode(text)
    if remove_control:
        text = remove_control_chars(text)
    if normalize_ws:
        text = normalize_whitespace(text)
    if fix_punct:
        text = fix_punctuation(text)
    if turkish_corrections:
        text = apply_turkish_corrections(text)
    return text.strip()


# ─────────────────────────────────────────────
# 3. OCRResult son işlemesi
# ─────────────────────────────────────────────

def postprocess_result(result: OCRResult, cfg: dict) -> OCRResult:
    """
    OCRResult nesnesine son işleme uygular.

    Args:
        result: Ham OCR sonucu.
        cfg: config.yaml'dan 'postprocessing' bloğu.

    Returns:
        Temizlenmiş OCRResult.
    """
    norm_cfg = cfg.get("normalization", {})
    tr_cfg = cfg.get("turkish_corrections", {})

    # Tam metni temizle
    result.full_text = normalize_text(
        result.full_text,
        fix_unicode_flag=norm_cfg.get("fix_unicode", True),
        remove_control=norm_cfg.get("remove_control_chars", True),
        normalize_ws=norm_cfg.get("normalize_whitespace", True),
        fix_punct=norm_cfg.get("fix_punctuation", True),
        turkish_corrections=tr_cfg.get("enabled", True)
        and tr_cfg.get("common_replacements", True),
    )

    # Kelime bazında da temizle
    for word in result.words:
        word.text = normalize_text(
            word.text,
            fix_unicode_flag=norm_cfg.get("fix_unicode", True),
            remove_control=norm_cfg.get("remove_control_chars", True),
            normalize_ws=False,
            fix_punct=False,
            turkish_corrections=tr_cfg.get("enabled", True),
        )

    logger.info("Son işleme tamamlandı.")
    return result


# ─────────────────────────────────────────────
# 4. Çıktı üretici: TXT
# ─────────────────────────────────────────────

def export_txt(
    result: OCRResult,
    output_path: str,
    include_metadata: bool = True,
    encoding: str = "utf-8",
) -> None:
    """
    OCR sonucunu düz metin (TXT) olarak kaydeder.

    Args:
        result: İşlenmiş OCR sonucu.
        output_path: Çıktı dosyası yolu (.txt).
        include_metadata: Üst veri başlığı eklensin mi?
        encoding: Dosya kodlaması.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    if include_metadata:
        lines += [
            f"# OCR Sonucu",
            f"# Tarih       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Dil         : {result.language}",
            f"# PSM         : {result.psm}",
            f"# Kelime Sayısı: {len(result.words)}",
            f"# Ort. Güven  : {result.avg_confidence:.1f}%",
            f"# Boyut (HxW) : {result.image_shape[0]}x{result.image_shape[1]}",
            "─" * 60,
            "",
        ]
    lines.append(result.full_text)

    with open(p, "w", encoding=encoding) as f:
        f.write("\n".join(lines))
    logger.info("TXT çıktı kaydedildi: %s", output_path)


# ─────────────────────────────────────────────
# 5. Çıktı üretici: JSON
# ─────────────────────────────────────────────

def export_json(
    result: OCRResult,
    output_path: str,
    include_words: bool = True,
    include_bboxes: bool = True,
    encoding: str = "utf-8",
) -> None:
    """
    OCR sonucunu JSON formatında kaydeder.

    Args:
        result: İşlenmiş OCR sonucu.
        output_path: Çıktı dosyası yolu (.json).
        include_words: Kelime listesi dahil edilsin mi?
        include_bboxes: Bounding box'lar dahil edilsin mi?
        encoding: Dosya kodlaması.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "language": result.language,
            "psm": result.psm,
            "avg_confidence": result.avg_confidence,
            "word_count": len(result.words),
            "image_shape": {"height": result.image_shape[0], "width": result.image_shape[1]},
        },
        "full_text": result.full_text,
    }

    if include_words:
        words_list = []
        for w in result.words:
            entry: Dict[str, Any] = {
                "text": w.text,
                "confidence": w.confidence,
                "block_num": w.block_num,
                "line_num": w.line_num,
                "page_num": w.page_num,
            }
            if include_bboxes:
                entry["bbox"] = {
                    "x": w.bbox[0], "y": w.bbox[1],
                    "w": w.bbox[2], "h": w.bbox[3],
                }
            words_list.append(entry)
        payload["words"] = words_list

    with open(p, "w", encoding=encoding) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("JSON çıktı kaydedildi: %s", output_path)


# ─────────────────────────────────────────────
# 6. Çıktı üretici: CSV
# ─────────────────────────────────────────────

def export_csv(
    result: OCRResult,
    output_path: str,
    include_bboxes: bool = True,
    encoding: str = "utf-8",
) -> None:
    """
    Her kelimeyi ayrı satır olarak CSV formatında kaydeder.

    Sütunlar:
      page_num, block_num, line_num, text, confidence,
      [x, y, w, h]  (include_bboxes=True ise)

    Args:
        result: İşlenmiş OCR sonucu.
        output_path: Çıktı dosyası yolu (.csv).
        include_bboxes: Koordinat sütunları eklensin mi?
        encoding: Dosya kodlaması.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["page_num", "block_num", "line_num", "text", "confidence"]
    if include_bboxes:
        fieldnames += ["x", "y", "w", "h"]

    with open(p, "w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for word in result.words:
            row: Dict[str, Any] = {
                "page_num": word.page_num,
                "block_num": word.block_num,
                "line_num": word.line_num,
                "text": word.text,
                "confidence": word.confidence,
            }
            if include_bboxes:
                row.update({"x": word.bbox[0], "y": word.bbox[1],
                            "w": word.bbox[2], "h": word.bbox[3]})
            writer.writerow(row)

    logger.info("CSV çıktı kaydedildi: %s", output_path)


# ─────────────────────────────────────────────
# 7. Toplu dışa aktarma
# ─────────────────────────────────────────────

def export_all(
    result: OCRResult,
    base_path: str,
    cfg: dict,
) -> Dict[str, str]:
    """
    Konfigürasyonda belirtilen tüm formatlarda çıktı üretir.

    Args:
        result: İşlenmiş OCR sonucu.
        base_path: Çıktı dosyasının uzantısız temel yolu
                   (örn. 'outputs/belge_001').
        cfg: config.yaml'dan 'output' bloğu.

    Returns:
        Format → dosya yolu eşlemesi.
    """
    formats = cfg.get("formats", ["txt", "json", "csv"])
    inc_conf = cfg.get("include_confidence", True)
    inc_bbox = cfg.get("include_bboxes", True)
    inc_meta = cfg.get("include_metadata", True)
    encoding = cfg.get("encoding", "utf-8")

    paths: Dict[str, str] = {}
    for fmt in formats:
        fmt = fmt.lower()
        out_path = f"{base_path}.{fmt}"
        try:
            if fmt == "txt":
                export_txt(result, out_path, include_metadata=inc_meta, encoding=encoding)
            elif fmt == "json":
                export_json(result, out_path, include_words=True,
                            include_bboxes=inc_bbox, encoding=encoding)
            elif fmt == "csv":
                export_csv(result, out_path, include_bboxes=inc_bbox, encoding=encoding)
            else:
                logger.warning("Desteklenmeyen format '%s'; atlandı.", fmt)
                continue
            paths[fmt] = out_path
        except Exception as exc:
            logger.error("'%s' formatında dışa aktarma hatası: %s", fmt, exc)

    return paths
