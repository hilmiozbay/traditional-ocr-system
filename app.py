"""
app.py
======
OCR Web Arayüzü - Flask backend.
Kullanıcı görüntü yükler → OCR boru hattından geçer → metin kopyalanabilir.
"""

import io
import sys
import os
import base64
import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image

# ── Modül yolunu ayarla ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "main_processes"))

try:
    from main_processes.preprocessing import preprocess_pipeline
    from main_processes.segmentation import segment_document
    from main_processes.ocr import run_ocr_pipeline, analyze_confidence
    from main_processes.postprocessing import postprocess_result
except ImportError:
    from main_processes.preprocessing import preprocess_pipeline
    from main_processes.segmentation import segment_document
    from main_processes.ocr import run_ocr_pipeline, analyze_confidence
    from main_processes.postprocessing import postprocess_result

# ── Flask uygulaması ─────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Konfigürasyon ─────────────────────────────────────────────
CFG_PATH = Path(__file__).parent / "config.yaml"
with open(CFG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

UPLOAD_DIR = Path(__file__).parent / "temp" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """PIL → OpenCV BGR dönüşümü."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_base64(img: np.ndarray, ext: str = ".png") -> str:
    """OpenCV görüntüsünü base64 string'e çevirir."""
    success, buf = cv2.imencode(ext, img)
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def draw_word_boxes(img: np.ndarray, words: list) -> np.ndarray:
    """
    Her kelime üzerine yarı saydam highlight kutusu çizer.
    Güven skoruna göre renk değişir:
      ≥ 80 → yeşil | 50-79 → sarı | < 50 → kırmızı
    """
    overlay = img.copy()
    for word in words:
        conf = word.confidence
        x, y, w, h = word.bbox
        if conf >= 80:
            color = (34, 197, 94)    # yeşil
        elif conf >= 50:
            color = (234, 179, 8)    # sarı
        else:
            color = (239, 68, 68)    # kırmızı
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

    # %30 saydamlıkla birleştir
    result = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    # Çerçeve çiz
    for word in words:
        conf = word.confidence
        x, y, w, h = word.bbox
        color = (34, 197, 94) if conf >= 80 else (234, 179, 8) if conf >= 50 else (239, 68, 68)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 1)
    return result


# ── Rotalar ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    """
    POST /ocr
    Form-data: file (görüntü dosyası), lang (isteğe bağlı), psm (isteğe bağlı)

    Döner:
    {
      "success": true,
      "text": "...",
      "confidence": 82.7,
      "word_count": 30,
      "elapsed": 1.23,
      "annotated_image": "<base64>",
      "words": [{"text":"...", "confidence":96, "bbox":[x,y,w,h]}, ...]
    }
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Dosya bulunamadı."}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"success": False,
                        "error": "Desteklenmeyen format. JPG, PNG, TIFF kullanın."}), 400

    lang = request.form.get("lang", CFG["ocr"]["tesseract"]["language"])
    psm  = int(request.form.get("psm",  CFG["ocr"]["tesseract"]["psm"]))

    t0 = time.perf_counter()
    try:
        # Dosyayı oku → PIL → OpenCV
        file_bytes = file.read()
        pil_img = Image.open(io.BytesIO(file_bytes))
        img = pil_to_cv2(pil_img)

        # Config'i güncelle
        cfg = dict(CFG)
        cfg["ocr"] = dict(cfg.get("ocr", {}))
        cfg["ocr"]["tesseract"] = dict(cfg["ocr"].get("tesseract", {}))
        cfg["ocr"]["tesseract"]["language"] = lang
        cfg["ocr"]["tesseract"]["psm"] = psm

        # ── Ön işleme ──
        proc_img, binary = preprocess_pipeline(img, cfg.get("preprocessing", {}))

        # ── Segmentasyon ──
        seg = segment_document(proc_img, binary, cfg.get("segmentation", {}))

        # ── OCR ──
        ocr_result = run_ocr_pipeline(
            proc_img, binary, cfg.get("ocr", {}),
            text_regions=seg["text_regions"]
        )
        clean_result = postprocess_result(ocr_result, cfg.get("postprocessing", {}))
        stats = analyze_confidence(clean_result)

        # ── Annotated görüntü (kelimeleri kutula) ──
        annotated = draw_word_boxes(proc_img, clean_result.words)
        # Görüntüyü orijinal boyutuna göre yeniden ölçekle (max 900px genişlik)
        h, w = annotated.shape[:2]
        if w > 900:
            scale = 900 / w
            annotated = cv2.resize(annotated,
                                   (900, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)

        annotated_b64 = cv2_to_base64(annotated)
        elapsed = round(time.perf_counter() - t0, 2)

        return jsonify({
            "success": True,
            "text": clean_result.full_text,
            "confidence": round(clean_result.avg_confidence, 1),
            "word_count": stats["word_count"],
            "elapsed": elapsed,
            "annotated_image": annotated_b64,
            "words": [
                {
                    "text": w_.text,
                    "confidence": round(w_.confidence, 1),
                    "bbox": list(w_.bbox),
                    "line": w_.line_num,
                }
                for w_ in clean_result.words
            ],
            "segments": {
                "lines":   len(seg["text_lines"]),
                "columns": len(seg["columns"]),
                "tables":  len(seg["tables"]),
            }
        })

    except Exception as e:
        logger.exception("OCR hatası")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  OCR Web Arayüzü başlatılıyor...")
    print("  Tarayıcıda aç: http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5000)
