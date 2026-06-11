# **End-to-End RAG System with MLOps Tracking**

> **Studi Kasus:** Sistem Tanya-Jawab Otomatis Dokumen SOP Deteksi Cacat Tekstil Pabrik berbasis LLM Lokal.

---

## **1. OVERVIEW**
Project ini mengimplementasikan sistem *Retrieval-Augmented Generation* (RAG) tingkat produksi yang diintegrasikan dengan prinsip MLOps. Sistem ini dirancang untuk menjawab pertanyaan seputar dokumen *Standard Operating Procedure* (SOP) deteksi cacat pada industri tekstil (seperti noda minyak, putus benang, dan belang warna) secara akurat tanpa halusinasi. Melalui pendekatan MLOps, proyek ini tidak hanya berfokus pada pembuatan chatbot, melainkan pada siklus eksperimen, pelacakan metrik, otomatisasi database vektor, dan kontainerisasi untuk memastikan sistem dapat dipantau dan direproduksi dengan mudah.

## **2. ALUR PENGERJAAN**
Project ini dibangun melalui beberapa fase pengembangan yang terstruktur:

1. Analisis Dokumen dan *Parsing*: Dokumen dibaca dan dibersihkan dari karakter yang tidak diperlukan.

2. *Ingestion* dan *Text Chunking*: Teks dipotong-potong menjadi beberapa kepingan (chunks) menggunakan parameter ukuran (`chunk size`) dan tumpang-tindih (`overlap`) tertentu. Proses ini krusial untuk menjaga keutuhan konteks kalimat.

3. *Embedding* dan *Vector Storage*: Kepingan teks diubah menjadi vektor numerik menggunakan model embedding lalu disimpan ke dalam database vektor lokal (**ChromaDB**). Setiap kali proses sinkronisasi dijalankan, database lama akan dihapus bersih (reset) otomatis agar data tidak duplikat atau tumpang tindih.

4. *Retrieval* dan *Top-K Filtering*: Saat user mengajukan pertanyaan, sistem mencari kepingan teks yang paling relevan secara semantik berdasarkan parameter K teratas (Top-K).

5. *Generation* via Local LLM: Kepingan teks yang berhasil ditarik kemudian dikirimkan sebagai konteks tambahan ke LLM (**Qwen2.5-3B**) untuk menghasilkan jawaban yang presisi dan berbasis fakta.

6. *MLOps Logging* dan *Tracking*: Setiap variasi hyperparameter (`chunk size`, `overlap`, `Top-K`) beserta metrik hasil eksekusinya (`total chunk`, `waktu ingestion`, `kualitas jawaban`) dicatat secara otomatis ke MLflow.

## **3. TECH STACKS**
Komponen teknologi yang digunakan dalam arsitektur ini meliputi:

* **Bahasa Pemrograman**: `Python`

* **Framework Backend API**: `FastAPI` dan `Uvicorn`

* **Antarmuka Pengguna**: `Streamlit`

* **Database Vektor**: `ChromaDB`

* **Model Embedding**: `all-MiniLM-L6-v2` (via HuggingFace/LangChain)

* **LLM**: `Qwen2.5:3b` (dijalankan secara lokal menggunakan Ollama)

* **MLOps**: `MLflow`

* **Lingkungan Pengembangan & Deployment**: `Docker`, `Docker Compose`, dan `Windows Subsystem for Linux (WSL)`

## **4. EVALUASI DAN HYPERPARAMETER SWEEPING**
Untuk menemukan konfigurasi terbaik, pada project ini dilakukan pengujian terukur (*Grid Search*) melalui 3 fase eksperimen dengan 6 variasi eksekusi (runs) yang dipantau langsung via MLflow:

**Eksperimen 1 (*Micro-Chunking*)**: Menggunakan ukuran kecil (**`Chunk size = 150 & 200`**). Hasilnya model kehilangan makna fungsional karena kalimat SOP terputus-putus, sehingga LLM tidak mampu memberikan detail informasi.

**Eksperimen 2 (*The Sweet Spot*)**: Menggunakan ukuran menengah (**`Chunk size = 350 & 500`**). Terjadi fenomena halusinasi parsial pada ukuran 350 karena informasi terpotong di tengah poin daftar, sehingga LLM meralat jawabannya sendiri akibat data yang tidak utuh.

**Eksperimen 3 (*Macro-Chunking*)**: Menggunakan ukuran besar (**`Chunk size = 800 & 1000`**). Konfigurasi `Chunk Size = 800`, `Overlap = 50`, dan `Top-K = 3` dipilih sebagai arsitektur final terbaik. Pilihan ini didasarkan pada data MLflow yang menunjukkan latensi penulisan terendah (karena efisiensi batch embedding hanya menghasilkan 2 chunks) serta memberikan akurasi jawaban 100 persen tanpa halusinasi pada model Qwen.

Lebih lengkapnya, dapat dilihat pada tabel berikut.

| Skenario | Parameter (Size/Overlap/k) | Karakteristik Jawaban | Status | Analisis Fenomena MLOps |
|---------------|----------------------------|-----------------------------------------------------------------------|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Eksperimen 1A | 150 / 20 / 8               | Tahu ada 3 cacat, tapi gagal menyebutkan detailnya.                   | Gagal (Parsial) | Information Fragmentation: Teks terlalu sempit sehingga kalimat pengantar terputus dari poin detail. Vektor gagal menyatukan kembali teks yang hancur berkeping-keping.                                        |
| Eksperimen 1B | 200 / 30 / 10              | Berhasil menjabarkan 3 cacat lengkap dengan detailnya.                | Akurat          | High-K Compensation: Walau teksnya terpotong kecil, nilai k=10 yang tinggi "memaksa" sistem menarik hampir semua kepingan yang relevan ke dalam otak LLM.                                                      |
| Eksperimen 2A | 350 / 50 / 4               | Menyebut 3 cacat di awal, tapi hanya menjabarkan 2, dan meralat diri. | Halusinasi      | Boundary Truncation: Ukuran 350 pas untuk menampung judul dan 2 poin pertama, tapi poin ke-3 terpotong ke chunk berikutnya dan tidak ikut terambil (karena k=4). LLM bingung melihat ketidakcocokan informasi. |
| Eksperimen 2B | 500 / 75 / 6               | Sangat rapi, terstruktur, poin presisi tinggi.                        | Sangat Sempurna | The Sweet Spot: Ukuran 500 sangat ideal membungkus satu klaster informasi prosedur dalam bahasa Indonesia secara utuh. Dengan k=6, seluruh spektrum SOP tersaji tanpa kepotongan.                                  |
| Eksperimen 3A | 800 / 100 / 3              | Akurat, lengkap, dan menyertakan regulasi dengan baik.                | Akurat          | Macro-Chunking: Dokumen hanya menjadi 2-4 bagian besar. Dengan k=3, hampir seluruh dokumen masuk sekaligus. Risiko salah ambil informasi berkurang drastis.                                                    |
| Eksperimen 3B | 1000 / 150 / 2             | Jawaban sangat lengkap karena membaca dokumen seutuhnya.              | Akurat          | Full-Context Ingestion: LLM membaca seolah-olah file utuh. Sangat aman untuk dokumen yang masih sedikit, namun jika dokumen sudah ada ratusan, metode ini akan memakan banyak memori (bottleneck).             |

## **5. LANGKAH LANGKAH INSTALASI DAN EKSEKUSI LOKAL**
Ikuti urutan langkah berikut untuk menjalankan seluruh ekosistem aplikasi di lingkungan lokal WSL Anda:

### Langkah 1: Persiapan Awal dan Ollama

1. Pastikan sudah berada di dalam terminal WSL.

2. Instal Ollama dengan menjalankan perintah:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

3. Unduh model LLM lokal dengan menjalankan perintah: 

```bash
ollama pull qwen2.5:3b
```

### Langkah 2: Sinkronisasi Kode dan Dependensi

1. Masuk ke dalam direktori utama project.

2. Buat environment virtual Python (opsional namun direkomendasikan) dan aktifkan. Bisa menggunakan `python` atau `conda`.

3. Instal seluruh library yang dibutuhkan seperti `fastapi`, `uvicorn`, `streamlit`, `mlflow`, dsb. Library tersebut dapat disimpan ke dalam file `requirements.txt`, lalu instal dengan menjalankan perintah:
```bash
pip install -r requirements.txt
```

### Langkah 3: Konfigurasi Parameter Lingkungan

1. Buka file `config.py`.

2. Atur alamat `OLLAMA_BASE_URL` mengarah ke `http://127.0.0.1:11434`.

3. Tentukan nama eksperimen pada variabel `MLFLOW_EXPERIMENT_NAME`, misalnya `RAG_Grid_Search_SOP`.

4. Taruh dokumen SOP tekstil atau dokumen lain yang ingin diuji ke dalam folder `data/documents`.

### Langkah 4: Menjalankan Ekosistem Aplikasi

1. Buka 3 tab terminal WSL baru dan jalankan perintah berikut secara terpisah pada masing-masing terminal:

    * **Terminal 1 (Dashboard MLflow)**: 
    ```bash
    mlflow ui --port 5000
    ```

    * **Terminal 2 (Backend API)**: 
    ```bash
    uvicorn server:app --port 8000 --reload
    ```

    * **Terminal 3 (Frontend Streamlit)**: 
    ```bash
    streamlit run app.py
    ```

### Langkah 5: Cara Melakukan Eksperimen Grid Search

1. Buka file `rag_engine.py`, sesuaikan nilai `CHUNK_SIZE`, `CHUNK_OVERLAP`, dan `Top-K` (misal untuk awal masukkan angka 150, 20, dan 8).

2. Simpan file tersebut. Terminal Backend akan otomatis memuat ulang perubahan kode.

3. Buka browser dan akses Streamlit dengan URL `localhost:8501`.

4. Klik tombol `Sync dan Update Database Vektor` untuk membersihkan database lama dan memproses dokumen dengan parameter baru.

5. Ajukan pertanyaan evaluasi standar pada kolom chat, contoh jika dokumennya berupa ***SOP deteksi cacat pada tekstil***: `Apa saja kategori cacat yang dapat dideteksi?`

6. Buka dashboard MLflow di `localhost:5000` untuk melihat, membandingkan grafik batang `total_chunks`, dan menganalisis latensi dari setiap parameter yang diuji.

7. Ulangi langkah di atas dengan kombinasi angka optimal hasil evaluasi (`Chunk 800`, `Overlap 50`, `Top-K 3`) untuk deployment akhir.

## **6. Catatan Pengembangan Lanjutan: Evaluasi dan Optimasi Otomatis**

* Untuk tahap pengembangan skala produksi di masa mendatang, proses pencarian kombinasi hyperparameter terbaik seperti `ukuran chunk`, `overlap`, dan `top_k` tidak perlu lagi dilakukan secara manual. Kita dapat mengotomatisasi proses evaluasi dan pencarian parameter ini menggunakan framework khusus evaluasi RAG seperti `Ragas` atau `TruLens`.

* Dengan mengintegrasikan framework tersebut ke dalam pipeline MLOps yang sudah dibangun, sistem dapat secara otomatis menilai setiap eksperimen kombinasi parameter berdasarkan metrik matematis yang objektif. 
* Metrik seperti `tingkat presisi konteks`, `relevansi pencarian`, dan `deteksi halusinasi model` dapat diukur secara algoritma. Hal ini akan menggantikan proses `grid search` manual menjadi siklus optimasi yang sepenuhnya otomatis, sehingga pencarian arsitektur RAG paling optimal menjadi jauh lebih efisien untuk menangani database dokumen yang sangat besar.