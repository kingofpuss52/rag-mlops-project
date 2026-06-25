import time
import gc
import mlflow
from rag_engine import ingest_documents, get_rag_chain
import config

def uji_kualitatif(chunk_size, chunk_overlap, top_k, nama_uji):
    # Set nama eksperimen KHUSUS untuk fase validasi
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("RAG_MLOps_Testing")
    
    with mlflow.start_run(run_name=nama_uji):
        # 1. Catat parameter yang sedang diuji
        mlflow.log_params({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k
        })
        
        # 2. Reset dan potong dokumen sesuai parameter saat ini
        print(f"\n🔄 Menyiapkan Database untuk [{nama_uji}]...")
        ingest_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # 3. Bangun chain dan siapkan pertanyaan
        chain = get_rag_chain(top_k=top_k)
        pertanyaan = "Bagaimana penanganan error jika sistem terlalu sering mendeteksi cacat dalam kurun waktu 1 jam?"
        
        print(f"🔮 Menanyakan: '{pertanyaan}'")
        start_time = time.time()
        
        # 4. Eksekusi inferensi
        jawaban = chain.invoke(pertanyaan)
        durasi = time.time() - start_time
        
        # 5. Catat teks jawaban lengkap dan kecepatan latensi ke MLflow
        mlflow.log_text(jawaban, "teks_jawaban_qwen_uji.txt")
        mlflow.log_metric("inference_latency_seconds", durasi)
        
        print("\n🤖 Jawaban Model:")
        print(jawaban)
        print(f"⏱️ Waktu Inferensi: {durasi:.2f} detik")
        print("-" * 50)
        
        # bersihkan database untuk pengujian berikutnya
        del chain
        gc.collect()
        time.sleep(3)

if __name__ == "__main__":
    # Tinggal masukkan variasi angka di dalam rentang optimal 
    # chunk_size = 400-600, chunk_overlap = 40-60, top_k = 4-7
    
    # uji_kualitatif(chunk_size=400, chunk_overlap=40, top_k=7, nama_uji="Uji 1 (cs = 400, co = 40, top_k = 7)")
    
    # uji_kualitatif(chunk_size=500, chunk_overlap=50, top_k=6, nama_uji="Uji 2 (cs = 500, co = 50, top_k = 6)")
    
    uji_kualitatif(chunk_size=600, chunk_overlap=60, top_k=4, nama_uji="Uji 3 (cs = 600, co = 60, top_k = 4)")
    