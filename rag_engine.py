import os
import time
from pathlib import Path
import mlflow
import shutil

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama

# Menggunakan core utilities (ringan dan stabil)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import config

# Set URI tempat MLflow mencatat data
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

def get_embedding_function():
    """Menginisialisasi model embedding HuggingFace."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

def ingest_documents():
    """Membaca dokumen dari folder data/documents, membuat embedding, dan menyimpannya ke ChromaDB."""
    # Konfigurasi parameter chunking (bisa diubah-ubah untuk eksperimen)
    CHUNK_SIZE = 150
    CHUNK_OVERLAP = 20
    
    # mulai catat eksperimen di mlflow
    with mlflow.start_run(run_name="ingestion_pipeline"):
      # log parameter ke mlflow
      mlflow.log_param("chunk_size", CHUNK_SIZE)    
      mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)    
      mlflow.log_param("embedding_model", config.EMBEDDING_MODEL)
      
      # mulai hitung waktu (latency)
      start_time = time.time()
      
      # Ini memastikan metadata dan jumlah chunk yang tercatat adalah murni dari parameter saat ini
      db_path = Path(config.DB_DIR)
      if db_path.exists():
          shutil.rmtree(db_path)
          
      # load dokumen
      loader = DirectoryLoader(str(config.DOCS_DIR), glob="**/*.txt", loader_cls=TextLoader)
      documents = loader.load()
      
      if not documents:
          return "Tidak ada dokumen berformat .txt yang ditemukan di folder data/documents."
      
      # Proses Chunking
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
      chunks = text_splitter.split_documents(documents)
      
      # Simpan ke ChromaDB lokal
      embeddings = get_embedding_function()
      Chroma.from_documents(
          documents=chunks, 
          embedding=embeddings, 
          persist_directory=str(config.DB_DIR)
      )
      
      # hitung durasi proses dalam detik
      execution_time = time.time() - start_time
      
      # log metric ke mlflow
      mlflow.log_metric("total_documents", len(documents))
      mlflow.log_metric("total_chunks", len(chunks))
      mlflow.log_metric("ingestion_latency_time", execution_time)
      
      # log artifact (menyimpan salinan dokumen ke server mlflow)
      for doc in config.DOCS_DIR.glob("**/*.txt"):
        mlflow.log_artifact(str(doc), artifact_path="source_documents")
      
      return f"Berhasil memproses {len(chunks)} potongan teks dari {len(documents)} dokumen. (Dicatat oleh MLflow)"

def format_docs(docs):
    """Menggabungkan potongan-potongan dokumen menjadi satu teks utuh untuk konteks LLM."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    """Membangun RAG pipeline menggunakan operator LCEL murni (|)."""
    embeddings = get_embedding_function()
    
    # Memuat database dari path persisten
    db = Chroma(persist_directory=str(config.DB_DIR), embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 8})
    
    # Menghubungkan ke Ollama via Docker
    # llm = Ollama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
    llm = Ollama(model=config.LLM_MODEL)
    
    # Membuat Prompt Template
    system_prompt = (
        "Anda adalah asisten pintar. Jawablah pertanyaan pengguna hanya berdasarkan konteks yang diberikan di bawah ini.\n"
        "Jika Anda tidak tahu jawabannya atau tidak ada di dokumen, katakan bahwa informasi tidak ditemukan di dokumen.\n\n"
        "Konteks:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # PIPELINE LCEL MURNI (Menggantikan create_retrieval_chain sepenuhnya)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser() # Otomatis mengubah output LLM langsung menjadi string teks biasa
    )
    
    return rag_chain