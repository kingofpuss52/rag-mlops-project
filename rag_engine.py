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
# mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
# mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

def get_embedding_function():
    """Menginisialisasi model embedding HuggingFace."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

def ingest_documents(chunk_size=config.DEFAULT_CHUNK_SIZE, chunk_overlap=config.DEFAULT_CHUNK_OVERLAP):
    """Membaca dokumen dari folder data/documents, membuat embedding, dan menyimpannya ke ChromaDB."""
    # Konfigurasi parameter chunking (bisa diubah-ubah untuk eksperimen)
    # CHUNK_SIZE = chunk_size
    # CHUNK_OVERLAP = chunk_overlap
    
    # mulai catat eksperimen di mlflow
    # with mlflow.start_run(run_name="ingestion_pipeline", nested=True):
    # log parameter ke mlflow
    # mlflow.log_param("chunk_size", CHUNK_SIZE)    
    # mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)    
    # mlflow.log_param("embedding_model", config.EMBEDDING_MODEL)
    
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
    # mlflow.log_metric("total_documents", len(documents))
    # mlflow.log_metric("total_chunks", len(chunks))
    # mlflow.log_metric("ingestion_latency_time", execution_time)
    
    # log artifact (menyimpan salinan dokumen ke server mlflow)
    # for doc in config.DOCS_DIR.glob("**/*.txt"):
    #     mlflow.log_artifact(str(doc), artifact_path="source_documents")
    
    # akhiri run
    # mlflow.end_run()
    
    return f"Berhasil memproses {len(chunks)} potongan teks dari {len(documents)} dokumen. (Dicatat oleh MLflow)"

def format_docs(docs):
    """Menggabungkan potongan-potongan dokumen menjadi satu teks utuh untuk konteks LLM."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(top_k=config.DEFAULT_TOP_K):
    """Membangun RAG pipeline menggunakan operator LCEL murni (|)."""
    embeddings = get_embedding_function()
    
    # Memuat database dari path persisten
    db = Chroma(persist_directory=str(config.DB_DIR), embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    
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

def get_answer_rag(question: str):
    """
    Fungsi pembungkus untuk mengambil jawaban nyata dari LLM 
    sekaligus mengembalikan kepingan dokumen asli (chunks) untuk Ragas.
    """
    embeddings = get_embedding_function()
    db = Chroma(persist_directory=str(config.DB_DIR), embedding_function=embeddings)
    
    # 1. Ambil dokumen relevan secara manual terlebih dahulu (k=8 sesuai settinganmu)
    retriever = db.as_retriever(search_kwargs={"k": 8})
    docs_relevan = retriever.invoke(question)
    
    # 2. Ekstrak isi teks teks potongan ke dalam list of strings untuk Ragas
    source_documents = [doc.page_content for doc in docs_relevan]
    
    # 3. Jalankan chain Ragas seperti biasa
    chain = get_rag_chain()
    llm_answer = chain.invoke(question)
    
    del db, retriever, chain
    
    # 4. Kembalikan dalam bentuk dictionary sesuai kebutuhan skrip evaluasi
    return {
        "answer": llm_answer,
        "source_documents": source_documents
    }