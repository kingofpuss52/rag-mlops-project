# Menggunakan image Python yang ringan sebagai dasar
FROM python:3.10-slim

# Mengatur folder kerja di dalam container
WORKDIR /app

# Menginstal dependensi sistem yang dibutuhkan oleh ChromaDB dan g++
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Menyalin file requirements dan menginstal library Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode proyek ke dalam container
COPY . .

# Expose port yang sekiranya akan digunakan
EXPOSE 8501 5000 8000