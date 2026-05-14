# ── 1. Aşama: Tesseract + sistem bağımlılıkları ──────────────
FROM python:3.11-slim

# Tesseract + Türkçe dil paketi + OpenCV sistem kütüphaneleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-tur \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Aşama: Python bağımlılıkları ─────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 3. Aşama: Uygulama dosyaları ─────────────────────────────
COPY . .

# Geçici upload klasörünü oluştur
RUN mkdir -p temp/uploads outputs

# Tesseract yolunu ortam değişkeniyle belirt
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

EXPOSE 5000

# Gunicorn ile production sunucu
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
