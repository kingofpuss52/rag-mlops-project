import os
from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
import pandas as pd

from rag_engine import get_answer_rag, ingest_documents

# 1. Definisikan Kumpulan Pertanyaan Uji dan Kunci Jawaban (Ground Truth)
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
    data_ingest = ingest_documents(chunk_size=800, chunk_overlap=50)
    print(data_ingest)
    
    print("🚀 Inisialisasi Model Hakim (Judge) via ChatOllama Lokal...")
    
    # 2. Inisialisasi Model Bahasa dan Embedding menggunakan paket resmi terbaru
    llm_lokal = ChatOllama(model="qwen2.5:3b", base_url="http://127.0.0.1:11434", temperature=0)
    embedding_lokal = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Bungkus model ke dalam struktur Wrapper Ragas
    ragas_llm = LangchainLLMWrapper(llm_lokal)
    ragas_emb = LangchainEmbeddingsWrapper(embedding_lokal)
    
    # Siapkan metrik dan pasangkan juri lokal secara eksplisit
    metrik_evaluasi = [faithfulness, AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    for metric in metrik_evaluasi:
        metric.llm = ragas_llm
        if hasattr(metric, 'embeddings'):
            metric.embeddings = ragas_emb

    print("📬 Mengumpulkan jawaban simulasi...")
    answers = []
    contexts = []
    
    # ambil data real
    for q in test_dataset["question"]:
        print(f"-> Memproses pertanyaan: {q}")
        
        # panggil fungsi backend
        real_response = get_answer_rag(q)
        
        answers.append(real_response["answer"])
        contexts.append(real_response["source_documents"])
    
    ragas_data = {
        "question": test_dataset["question"],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": test_dataset["ground_truth"]
    }
    
    dataset = Dataset.from_dict(ragas_data)
    
    print("📊 Memulai perhitungan skor metrik via Qwen2.5:3b...")
    try:
        score = evaluate(
            dataset=dataset,
            metrics=metrik_evaluasi
        )
        df_hasil = score.to_pandas()
        print("\n✅ Hasil Evaluasi Sukses!")
        print(df_hasil.to_string())
        
        # jadikan json
        nama_file_json = "log_evaluasi_ragas.json"
        df_hasil.to_json(
            nama_file_json,
            orient="records",
            indent=4,
            force_ascii=False
        )
        
        print(f"\n✅ Log Evaluasi Berhasil Disimpan ke dalam file {nama_file_json}!")
        
        return df_hasil
        
    except Exception as e:
        print(f"❌ Gagal mengevaluasi: {str(e)}")
        return None

if __name__ == "__main__":
    jalankan_eval_otomatis()