import os
from pathlib import Path

# Mendeteksi folder utama (root) dari proyek ini secara dinamis
BASE_DIR = Path(__file__).resolve().parent

# arahkan ke folder data internal
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"
DB_DIR = DATA_DIR / "vector_db"

# pastikan folder-folder tersebut sudah terbuat di sistem
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# Konfigurasi Model
LLM_MODEL = "qwen2.5:3b"  # Sesuaikan dengan model Ollama yang kamu unduh
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Konfigurasi MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "RAG_Dokumen_Internal"

# Mengarahkan LLM Ollama ke host WSL dari dalam Docker
# OLLAMA_BASE_URL = "http://host.docker.internal:11434"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Konfigurasi parameter hasil tuning dan uji
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 6