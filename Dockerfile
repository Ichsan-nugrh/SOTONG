# 1. Gunakan Python 3.11-slim (Stabil untuk TF & PyTorch)
FROM python:3.11-slim

# 2. Set Environment Variables
# PYTHONDONTWRITEBYTECODE: Mencegah Python menulis file .pyc
# PYTHONUNBUFFERED: Memastikan log langsung keluar di terminal
# TF_ENABLE_ONEDNN_OPTS: Mematikan warning oneDNN dari TensorFlow agar log bersih
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0

# 3. Instal dependensi sistem untuk OpenCV (cv2)
# Docker versi slim tidak memiliki library ini secara bawaan
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Tentukan direktori kerja di dalam container
WORKDIR /app

# 5. Salin requirements.txt terlebih dahulu (untuk optimasi Docker Caching)
# Jika requirements.txt tidak berubah, Docker tidak akan menginstal ulang TF & PyTorch
COPY requirements.txt .

# 6. Instal semua dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# 7. Salin seluruh sisa file proyek (streamlit_app.py, folder models, dll)
COPY . .

# 8. Ekspos port yang digunakan Streamlit
EXPOSE 8501

# 9. Perintah untuk menjalankan aplikasi
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]