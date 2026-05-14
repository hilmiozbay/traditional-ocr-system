# Geleneksel Belge Tarama ve OCR Sistemi

Python tabanlı, **derin öğrenme kullanmayan** belge tarama ve OCR sistemi.  
OpenCV · Tesseract · scikit-image · PyMuPDF · scikit-learn

---

## Proje Yapısı

```
goruntu_isleme_projesi/
├── config.yaml            ← Tüm parametreler burada
├── main.py                ← CLI giriş noktası
├── preprocessing.py       ← Görüntü ön işleme boru hattı
├── segmentation.py        ← Belge analizi & bölütleme
├── ocr.py                 ← Tesseract OCR motoru
├── postprocessing.py      ← Metin temizleme & çıktı üretimi
├── requirements.txt
├── outputs/               ← Çıktı dosyaları (otomatik oluşturulur)
├── temp/
├── sample_images/
└── tests/
    ├── test_preprocessing.py
    ├── test_segmentation.py
    └── test_postprocessing.py
```

---

## Kurulum

### 1. Tesseract OCR

Windows için [buradan](https://github.com/UB-Mannheim/tesseract/wiki) indirin.  
Kurulum sırasında **"Türkçe (tur)"** dil paketini seçin.

Kurulum sonrası `tesseract` komutunun PATH'te olduğunu doğrulayın:
```powershell
tesseract --version
```

### 2. Python bağımlılıkları

```powershell
pip install -r requirements.txt
```

---

## Kullanım

```powershell
# Tek görüntü
python main.py --input sample_images/belge.jpg

# Türkçe, belirli PSM ile
python main.py --input belge.png --lang tur --psm 6

# Klasördeki tüm belgeler
python main.py --input sample_images/ --output ciktilar/

# PDF dosyası
python main.py --input rapor.pdf --output ciktilar/

# Debug modu (ara görüntüler kaydedilir)
python main.py --input belge.jpg --debug

# Eğim düzeltmeyi kapat
python main.py --input belge.jpg --no-deskew
```

---

## Bileşenler

### `preprocessing.py`
| Fonksiyon | Açıklama |
|-----------|----------|
| `load_image()` | Türkçe yol desteğiyle görüntü yükleme |
| `resize_image()` | En-boy korumalı boyutlandırma |
| `denoise()` | Gaussian / Medyan / Bilateral filtre |
| `enhance_contrast()` | CLAHE veya histogram eşitleme |
| `binarize()` | Otsu / Adaptif Gaussian eşikleme |
| `deskew()` | Hough veya projeksiyon ile eğim düzeltme |
| `four_point_transform()` | Perspektif düzeltme |
| `preprocess_pipeline()` | Tüm adımları config'e göre çalıştırır |

### `segmentation.py`
| Fonksiyon | Açıklama |
|-----------|----------|
| `connected_component_analysis()` | Alan/oran filtrelemeli bileşen analizi |
| `detect_text_regions_mser()` | MSER ile metin bölgesi tespiti |
| `detect_text_lines()` | Yatay projeksiyon ile satır tespiti |
| `detect_columns()` | Dikey projeksiyon ile sütun tespiti |
| `detect_tables()` | Morfolojik yöntemle tablo tespiti |

### `ocr.py`
| Fonksiyon | Açıklama |
|-----------|----------|
| `build_tesseract_config()` | PSM/OEM/DPI config dizesi üretir |
| `prepare_for_ocr()` | Unsharp mask + binary hazırlık |
| `ocr_full_page()` | Tam sayfa OCR + WordResult listesi |
| `ocr_region()` | Tek bölge kırpıp OCR |
| `ocr_regions_batch()` | Birden fazla bölge için toplu OCR |
| `analyze_confidence()` | Güven istatistikleri |

### `postprocessing.py`
| Fonksiyon | Açıklama |
|-----------|----------|
| `normalize_text()` | Unicode + boşluk + noktalama normalleştirme |
| `apply_turkish_corrections()` | OCR Türkçe hata düzeltmeleri |
| `export_txt()` | Düz metin çıktısı |
| `export_json()` | Yapılandırılmış JSON çıktısı |
| `export_csv()` | Kelime düzeyinde CSV çıktısı |
| `export_all()` | Tüm formatları tek seferde üretir |

---

## Testler

```powershell
# Tüm testleri çalıştır
pytest tests/ -v

# Tek modülü test et
pytest tests/test_preprocessing.py -v

# Kapsam raporu
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Konfigürasyon (`config.yaml`)

Tüm parametreler `config.yaml` içinde gruplandırılmıştır:
- `preprocessing` → Gürültü, eşikleme, eğim, perspektif
- `segmentation`  → MSER, projeksiyon, tablo
- `ocr`           → Tesseract dil, PSM, OEM, güven eşiği
- `postprocessing`→ Normalleştirme, Türkçe düzeltmeler
- `output`        → Format seçimi, bounding box, metadata
- `pdf`           → DPI, sayfa aralığı

---

## PSM Modu Rehberi

| PSM | Kullanım senaryosu |
|-----|-------------------|
| 3   | Standart belge (varsayılan) |
| 4   | Tek sütunlu metin |
| 6   | Tek metin bloğu (form, tablo hücresi) |
| 7   | Tek satır |
| 8   | Tek kelime |
| 11  | Seyrek/dağınık metin |

---

## Kısıtlamalar

- El yazısı belgelerde Tesseract başarımı düşük olabilir
- Çok düşük çözünürlüklü görüntülerde (<150 DPI) OCR kalitesi azalır
- MSER yöntemi yoğun dokulu arka planlarda fazla bölge üretebilir;
  bu durumda `min_area` değerini artırın

---

## Lisans

MIT
