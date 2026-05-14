"""
main.py
=======
Geleneksel Belge Tarama ve OCR Sistemi - Ana giriş noktası.

Kullanım:
  python main.py --input belge.jpg
  python main.py --input belgeler/ --output sonuclar/
  python main.py --input belge.pdf --lang tur --psm 3
  python main.py --input belge.jpg --debug --no-deskew
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import yaml

from main_processes.preprocessing import load_image, preprocess_pipeline, save_image
from main_processes.segmentation import segment_document, draw_regions
from main_processes.ocr import run_ocr_pipeline, analyze_confidence
from main_processes.postprocessing import postprocess_result, export_all

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Konfigürasyon yükleme
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """
    YAML konfigürasyon dosyasını yükler.

    Args:
        config_path: Konfigürasyon dosyasının yolu.

    Returns:
        Konfigürasyon sözlüğü.
    """
    p = Path(config_path)
    if not p.exists():
        logger.warning("Konfigürasyon bulunamadı: %s. Varsayılanlar kullanılıyor.", config_path)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    logger.debug("Konfigürasyon yüklendi: %s", config_path)
    return cfg


# ─────────────────────────────────────────────
# PDF desteği
# ─────────────────────────────────────────────

def pdf_to_images(pdf_path: str, cfg: dict) -> List:
    """
    PDF dosyasını sayfa görüntülerine dönüştürür.

    Args:
        pdf_path: PDF dosyasının yolu.
        cfg: 'pdf' konfigürasyon bloğu.

    Returns:
        BGR NumPy dizileri listesi (bir sayfa = bir eleman).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF yüklü değil. Lütfen: pip install pymupdf")

    import numpy as np

    dpi = cfg.get("dpi", 200)
    first_page = cfg.get("first_page", 1) - 1  # 0-indexed
    last_page = cfg.get("last_page", None)

    doc = fitz.open(pdf_path)
    total = doc.page_count
    end = total if last_page is None else min(last_page, total)

    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 DPI → hedef DPI
    for page_num in range(first_page, end):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img_array = img_array.reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        images.append(img_array)
        logger.debug("PDF sayfa %d/%d dönüştürüldü.", page_num + 1, total)

    doc.close()
    logger.info("PDF dönüşümü: %d sayfa.", len(images))
    return images


# ─────────────────────────────────────────────
# Tek görüntü işleme
# ─────────────────────────────────────────────

def process_single_image(
    img,
    cfg: dict,
    output_base: str,
    debug: bool = False,
) -> dict:
    """
    Tek bir görüntü üzerinde tam OCR boru hattını çalıştırır.

    Args:
        img: BGR NumPy dizisi.
        cfg: Tam konfigürasyon sözlüğü.
        output_base: Çıktı dosyası yolunun uzantısız tabanı.
        debug: True ise ara sonuçları kaydeder.

    Returns:
        {'text', 'confidence', 'word_count', 'output_files'} sözlüğü.
    """
    t0 = time.perf_counter()

    # 1. Ön işleme
    logger.info("━━ Ön işleme başlıyor...")
    proc_img, binary = preprocess_pipeline(img, cfg.get("preprocessing", {}))

    if debug:
        save_image(binary, f"{output_base}_debug_binary.png")
        logger.info("Debug: binary görüntü kaydedildi.")

    # 2. Segmentasyon
    logger.info("━━ Segmentasyon başlıyor...")
    seg = segment_document(proc_img, binary, cfg.get("segmentation", {}))

    if debug:
        vis = draw_regions(proc_img, seg["text_regions"], color=(0, 255, 0), label="T")
        vis = draw_regions(vis, seg["tables"], color=(255, 0, 0), thickness=3, label="TBL")
        save_image(vis, f"{output_base}_debug_segments.png")
        logger.info("Debug: segmentasyon görüntüsü kaydedildi.")

    # 3. OCR
    logger.info("━━ OCR başlıyor...")
    ocr_result = run_ocr_pipeline(
        proc_img, binary, cfg.get("ocr", {}),
        text_regions=seg["text_regions"],
    )

    # 4. Güven analizi
    conf_stats = analyze_confidence(ocr_result)
    logger.info(
        "Güven istatistikleri: ort=%.1f%% min=%.1f%% max=%.1f%% kelime=%d",
        conf_stats["avg"], conf_stats["min"], conf_stats["max"], conf_stats["word_count"],
    )

    # 5. Son işleme
    logger.info("━━ Son işleme başlıyor...")
    clean_result = postprocess_result(ocr_result, cfg.get("postprocessing", {}))

    # 6. Çıktı üretimi
    logger.info("━━ Çıktılar kaydediliyor: %s", output_base)
    out_paths = export_all(clean_result, output_base, cfg.get("output", {}))

    elapsed = time.perf_counter() - t0
    logger.info("✓ Tamamlandı: %.2f sn | %s", elapsed, output_base)

    return {
        "text": clean_result.full_text,
        "confidence": clean_result.avg_confidence,
        "word_count": len(clean_result.words),
        "output_files": out_paths,
        "elapsed_sec": round(elapsed, 3),
        "segments": {
            "lines": len(seg["text_lines"]),
            "columns": len(seg["columns"]),
            "tables": len(seg["tables"]),
        },
    }


# ─────────────────────────────────────────────
# Komut satırı argümanları
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geleneksel Belge Tarama ve OCR Sistemi",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Giriş dosyası (görüntü/PDF) veya klasör yolu.",
    )
    parser.add_argument(
        "--output", "-o", default="outputs",
        help="Çıktı dizini (varsayılan: outputs/).",
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml",
        help="Konfigürasyon dosyası (varsayılan: config.yaml).",
    )
    parser.add_argument(
        "--lang", "-l", default=None,
        help="Tesseract dili (örn. tur, eng, tur+eng). config.yaml'ı geçersiz kılar.",
    )
    parser.add_argument(
        "--psm", type=int, default=None,
        help="Tesseract PSM modu (0-13). config.yaml'ı geçersiz kılar.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Ara görüntüleri kaydet (binary, segmentasyon).",
    )
    parser.add_argument(
        "--no-deskew", action="store_true",
        help="Eğim düzeltmeyi devre dışı bırak.",
    )
    parser.add_argument(
        "--no-perspective", action="store_true",
        help="Perspektif düzeltmeyi devre dışı bırak.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Ana işlev
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Logging ayarla
    cfg = load_config(args.config)
    log_level = cfg.get("general", {}).get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Komut satırı argümanlarını cfg'ye uygula
    if args.lang:
        cfg.setdefault("ocr", {}).setdefault("tesseract", {})["language"] = args.lang
    if args.psm is not None:
        cfg.setdefault("ocr", {}).setdefault("tesseract", {})["psm"] = args.psm
    if args.no_deskew:
        cfg.setdefault("preprocessing", {}).setdefault("deskew", {})["enabled"] = False
    if args.no_perspective:
        cfg.setdefault("preprocessing", {}).setdefault("perspective", {})["enabled"] = False

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    supported = set(cfg.get("general", {}).get(
        "supported_formats",
        [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".pdf"]
    ))

    # Giriş dosyalarını topla
    if input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.suffix.lower() in supported]
    elif input_path.exists():
        files = [input_path]
    else:
        logger.error("Giriş bulunamadı: %s", args.input)
        sys.exit(1)

    if not files:
        logger.warning("İşlenecek dosya yok: %s", args.input)
        sys.exit(0)

    logger.info("İşlenecek dosya sayısı: %d", len(files))
    all_results = []

    for file_path in files:
        logger.info("═══ İşleniyor: %s", file_path.name)
        stem = file_path.stem

        try:
            if file_path.suffix.lower() == ".pdf":
                # PDF: çok sayfalı
                pages = pdf_to_images(str(file_path), cfg.get("pdf", {}))
                for page_idx, page_img in enumerate(pages, start=1):
                    out_base = str(output_dir / f"{stem}_sayfa{page_idx:03d}")
                    res = process_single_image(page_img, cfg, out_base, args.debug)
                    res["source"] = f"{file_path.name} (s.{page_idx})"
                    all_results.append(res)
            else:
                img = load_image(str(file_path))
                out_base = str(output_dir / stem)
                res = process_single_image(img, cfg, out_base, args.debug)
                res["source"] = file_path.name
                all_results.append(res)

        except Exception as exc:
            logger.error("İşleme hatası [%s]: %s", file_path.name, exc, exc_info=True)
            all_results.append({"source": file_path.name, "error": str(exc)})

    # Özet rapor
    print("\n" + "═" * 60)
    print("  OCR İŞLEM RAPORU")
    print("═" * 60)
    for r in all_results:
        src = r.get("source", "?")
        if "error" in r:
            print(f"  ✗ {src:40s} HATA: {r['error']}")
        else:
            print(
                f"  ✓ {src:40s} "
                f"güven={r['confidence']:.0f}%  "
                f"kelime={r['word_count']}  "
                f"({r['elapsed_sec']:.1f}s)"
            )
    print("═" * 60)
    print(f"  Toplam: {len(all_results)} dosya işlendi → '{output_dir}/'")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
