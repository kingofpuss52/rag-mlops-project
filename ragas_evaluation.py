import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
from datasets import Dataset
import pandas as pd

# 1. Siapkan Kumpulan Pertanyaan Uji dan Kunci Jawaban (Ground Truth)
# Sesuaikan isi ground_truth di bawah ini dengan teks asli dari dokumen SOP tekstilmu
test_dataset = {
    "question": [
        "Apa saja kategori cacat yang dapat dideteksi?",
        "Berapa batas toleransi untuk cacat putus benang?",
        "Bagaimana penanganan error jika sistem terlalu sering mendeteksi cacat dalam kurun waktu 1 jam?"
    ],
    "ground_truth": [
        "Kategori cacat yang dapat dideteksi ada tiga, yaitu Noda Minyak (Oil Stain), Putus Benang (Broken Yarn), dan Belang Warna (Color Mismatch).",
        "Batas toleransi untuk cacat putus benang adalah celah struktural pada pola rajutan yang melebihi 2 milimeter.",
        "Jika terjadi anomali atau deteksi cacat terlalu sering dalam kurun waktu 1 jam, sistem akan memicu alarm, menghentikan mesin rajut otomatis, dan mengirimkan notifikasi ke teknisi untuk retraining model atau kalibrasi ulang kamera dengan target akurasi 92%."
    ]
}

def jalankan_eval_otomatis():
    print("🚀 Memulai proses evaluasi Ragas...")
    
    # Di tahap ini, kita perlu mensimulasikan pemanggilan fungsi dari rag_engine.py
    # untuk mendapatkan kolom 'answer' dan 'contexts' dari masing-masing question di atas.
    
    answers = []
    contexts = []
    
    # TODO: Integrasikan dengan fungsi chat dari rag_engine.py kamu
    # Contoh jalannya logika:
    # for q in test_dataset["question"]:
    #     hasil_rag = query_rag_engine(q)
    #     answers.append(hasil_rag["answer"])
    #     contexts.append(hasil_rag["source_chunks"]) # list of strings
    
    # Untuk uji coba struktur awal, kita buat dummy data terlebih dahulu
    answers = [
        "Model bisa mendeteksi Noda Minyak, Putus Benang, dan Belang Warna.",
        "Batas toleransinya adalah jika celah pola rajutan melebihi 2 milimeter.",
        "Sistem akan membunyikan alarm dan menghentikan mesin otomatis untuk retraining."
    ]
    contexts = [
        ["SOP mendeteksi tiga jenis cacat: Noda Minyak, Putus Benang, dan Belang Warna."],
        ["Cacat putus benang memiliki batas toleransi celah struktural rajutan melebihi 2 milimeter."],
        ["Jika cacat terlalu sering terdeteksi dalam 1 jam, sistem memicu alarm dan stop mesin otomatis."]
    ]
    
    # 2. Konversi ke Format Dataset HuggingFace yang dibutuhkan Ragas
    ragas_data = {
        "question": test_dataset["question"],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": test_dataset["ground_truth"]
    }
    
    dataset = Dataset.from_dict(ragas_data)
    
    # 3. Eksekusi Evaluasi Menggunakan Ragas
    # Catatan: Ragas secara default membutuhkan koneksi ke OpenAI/LLM eksternal sebagai Judge.
    # Kita bisa mengonfigurasinya agar menggunakan Qwen lokal kita sebagai Judge nanti.
    print("📊 Menghitung skor metrik Ragas...")
    try:
        score = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevance, context_precision, context_recall]
        )
        df_hasil = score.to_pandas()
        print("\n✅ Hasil Evaluasi Sukses!")
        print(df_hasil)
        
        # Di tahap akhir nanti, nilai df_hasil ini akan kita kirim langsung ke MLflow
    except Exception as e:
        print(f"❌ Gagal mengevaluasi: {str(e)}")

if __name__ == "__main__":
    jalankan_eval_otomatis()