import sys
import argparse
import mlflow
import time
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from rag_engine import ingest_documents, get_answer_rag
import config

# Dataset Uji Standar (Ground Truth)
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

def run_single_trial(chunk_size, chunk_overlap, trial_number, top_k):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    print(f"\n=================== RUNNING TRIAL SYSTEM #{trial_number} ===================")
    print(f"🕵️ Parameter -> Chunk Size: {chunk_size} | Overlap: {chunk_overlap}")
    
    # catat waktu mulai
    start_run_time = time.time()
    
    # Buka run di MLflow secara mandiri
    with mlflow.start_run(run_name=f"trial_with_bayesian_{trial_number}"):
        mlflow.log_params({
            "tuning_method": "CLI Decoupled Loop",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k
        })
        
        # 1. Bangun ulang database secara segar
        print(f"🔄 Memotong ulang dokumen untuk trial {trial_number}...")
        ingest_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # 2. Ambil jawaban dari RAG engine
        print(f"📬 Mengumpulkan sampel jawaban dari Qwen untuk trial {trial_number}...")
        answers = []
        contexts = []
        for q in test_dataset["question"]:
            respons = get_answer_rag(q)
            answers.append(respons["answer"])
            contexts.append(respons["source_documents"])
            
        # 3. Siapkan Juri Ragas Lokal
        llm_lokal = ChatOllama(model="qwen2.5:3b", base_url="http://127.0.0.1:11434", temperature=0)
        embedding_lokal = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        ragas_llm = LangchainLLMWrapper(llm_lokal)
        ragas_emb = LangchainEmbeddingsWrapper(embedding_lokal)
        
        metrik_evaluasi = [faithfulness, AnswerRelevancy(), ContextPrecision(), ContextRecall()]
        for metric in metrik_evaluasi:
            metric.llm = ragas_llm
            if hasattr(metric, 'embeddings'):
                metric.embeddings = ragas_emb
                
        # 4. Hitung Skor Menggunakan Ragas
        print("📊 Menghitung nilai metrik...")
        dataset_formatted = Dataset.from_dict({
            "question": test_dataset["question"],
            "answer": answers,
            "contexts": contexts,
            "ground_truth": test_dataset["ground_truth"]
        })
        
        try:
            score = evaluate(dataset=dataset_formatted, metrics=metrik_evaluasi)
            df_hasil = score.to_pandas()
            
            avg_precision = df_hasil["context_precision"].mean()
            avg_recall = df_hasil["context_recall"].mean()
            
            # hitung durasi tuning
            duration = (time.time() - start_run_time) / 60
                        
            # Kirim metrik ke MLflow
            mlflow.log_metrics({
                "mean_context_precision": avg_precision,
                "mean_context_recall": avg_recall,
                "execution_time_minute": duration
            })
            
            print(f"✅ Trial {trial_number} Sukses! Mean Precision: {avg_precision:.4f}")
            print(f"✅ Durasi Waktu: {duration} menit.")
            
        except Exception as e:
            print(f"❌ Trial gagal akibat: {str(e)}")

if __name__ == "__main__":
    # Menggunakan argparse agar skrip bisa menerima parameter dari terminal Bash
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--chunk_overlap", type=int, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()
    
    run_single_trial(args.chunk_size, args.chunk_overlap, args.top_k, args.trial)