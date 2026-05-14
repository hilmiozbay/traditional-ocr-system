"""
diagnose_turkish.py
===================
Türkçe karakter sorunlarını tespit eden teşhis scripti.
Sorun 1: Tesseract PATH'te değil
Sorun 2: pytesseract tesseract_cmd ayarlanmamış
Sorun 3: Türkçe tessdata eksik/bozuk
Sorun 4: Windows terminal encoding (CP1252 vs UTF-8)
Sorun 5: OCR çıktısında Türkçe karakter bozulması
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("=" * 60)
print("  TURKCE KARAKTER TESHIS RAPORU")
print("=" * 60)

# ── 1. Tesseract PATH kontrolü ───────────────────────────────
print("\n[1] Tesseract PATH kontrolu:")
import shutil
tess_in_path = shutil.which("tesseract")
if tess_in_path:
    print(f"    OK  - PATH'te bulundu: {tess_in_path}")
else:
    print(f"    UYARI - Tesseract sistem PATH'inde degil.")
    print(f"    Sabit yol kullanilacak: {TESS_PATH}")

if not os.path.exists(TESS_PATH):
    print(f"    HATA - {TESS_PATH} bulunamadi!")
    sys.exit(1)
else:
    print(f"    OK  - Dosya mevcut: {TESS_PATH}")

# ── 2. pytesseract yapılandırması ────────────────────────────
print("\n[2] pytesseract ayari:")
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH
    ver = pytesseract.get_tesseract_version()
    print(f"    OK  - Tesseract surumu: {ver}")
except Exception as e:
    print(f"    HATA - {e}")
    sys.exit(1)

# ── 3. Türkçe tessdata ───────────────────────────────────────
print("\n[3] Turkce dil paketi:")
try:
    langs = pytesseract.get_languages()
    if "tur" in langs:
        print(f"    OK  - 'tur' paketi mevcut. Tum diller: {langs}")
    else:
        print(f"    HATA - 'tur' paketi eksik! Mevcut: {langs}")
        print("    Cozum: https://github.com/tesseract-ocr/tessdata/blob/main/tur.traineddata")
except Exception as e:
    print(f"    HATA - {e}")

# ── 4. Terminal encoding ─────────────────────────────────────
print("\n[4] Terminal encoding:")
print(f"    sys.stdout.encoding : {sys.stdout.encoding}")
print(f"    sys.getdefaultencoding: {sys.getdefaultencoding()}")
print(f"    locale: ", end="")
import locale
print(locale.getpreferredencoding(False))
turkce_test = "ğüşıöçĞÜŞİÖÇ"
print(f"    Turkce karakterler  : {turkce_test}")

# ── 5. Basit OCR testi (sentetik görüntü) ────────────────────
print("\n[5] Sentetik Turkce OCR testi:")
try:
    import numpy as np, cv2
    from PIL import Image, ImageDraw, ImageFont

    # Türkçe metin içeren test görüntüsü oluştur
    img_pil = Image.new("RGB", (500, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_pil)

    # Sistem fontu bul
    font = None
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]
    for fc in font_candidates:
        if os.path.exists(fc):
            try:
                font = ImageFont.truetype(fc, 22)
                break
            except Exception:
                pass

    test_text = "Görev: şehirde çalışıyor, öğreniyor."
    if font:
        draw.text((10, 20), test_text, fill=(0, 0, 0), font=font)
        print(f"    Font kullanildi: {fc}")
    else:
        draw.text((10, 20), test_text, fill=(0, 0, 0))
        print("    Varsayilan font kullanildi (kalite dusuk olabilir).")

    # PIL → NumPy → OCR
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    config = "--psm 7 --oem 1 -l tur"
    result = pytesseract.image_to_string(img_pil, config=config).strip()

    print(f"    Beklenen  : {test_text}")
    print(f"    Taninan   : {result}")

    # Türkçe karakterlerin tanınma oranı
    tr_chars = set("ğüşıöçĞÜŞİÖÇ")
    expected_tr = [c for c in test_text if c in tr_chars]
    recognized_tr = [c for c in result if c in tr_chars]
    oran = len(recognized_tr) / len(expected_tr) * 100 if expected_tr else 0
    print(f"    Turkce karakter tanima: {len(recognized_tr)}/{len(expected_tr)} ({oran:.0f}%)")

    if oran < 50:
        print("    UYARI: Turkce karakter tanima dusuk!")
    else:
        print("    OK: Turkce karakter tanima kabul edilebilir.")

except Exception as e:
    print(f"    HATA - {e}")

# ── 6. image.png üzerinde gerçek test ────────────────────────
print("\n[6] image.png uzerinde gercek OCR testi:")
IMAGE = "image.png"
if not os.path.exists(IMAGE):
    print(f"    ATLANDI - {IMAGE} bulunamadi.")
else:
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.open(IMAGE)

        configs = [
            ("tur",     "--psm 3 --oem 1"),
            ("tur+eng", "--psm 3 --oem 1"),
            ("tur",     "--psm 6 --oem 1"),
        ]
        for lang, cfg_str in configs:
            txt = pytesseract.image_to_string(
                pil_img, lang=lang, config=cfg_str
            ).strip()
            print(f"\n    [lang={lang}, {cfg_str}]")
            print(f"    {repr(txt[:120])}")
    except Exception as e:
        print(f"    HATA - {e}")

# ── 7. Özet ve Çözüm Önerileri ───────────────────────────────
print("\n" + "=" * 60)
print("  COZUM ONERILERI")
print("=" * 60)
print("""
  A) Tesseract PATH'e ekle (bir kerelik):
     [Sistem Ozellikleri > Ortam Degiskenleri > PATH]
     Ekle: C:\\Program Files\\Tesseract-OCR

  B) Python kodunda her zaman su satiri ekle:
     import pytesseract
     pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

  C) OCR'da Turkce icin en iyi config:
     config = '--psm 3 --oem 1 -l tur'
     # PSM 3 = otomatik sayfa segmentasyonu
     # OEM 1 = LSTM motoru (en iyi Turkce destegi)

  D) Windows terminali UTF-8 icin:
     $env:PYTHONUTF8=1  (PowerShell'de)
     veya: python -X utf8 script.py

  E) Goruntu kalitesi dusukse on isleme yogunlastir:
     - DPI artir (min 300 DPI)
     - Kontrast artir (CLAHE)
     - Binary esikleme (Otsu)
""")
