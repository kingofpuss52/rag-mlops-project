# app.py
import streamlit as st
import requests
import os

# Mengambil URL backend dari environment variable Docker (default ke localhost)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Lokal RAG Bot", layout="centered")
st.title("🤖 Local RAG Asisten (Dockerize Environment)")

# Sidebar untuk manajemen dokumen
with st.sidebar:
    st.header("🗂️ Manajemen Dokumen")
    st.write(f"Taruh file `.txt` kamu di folder lokal, lalu sinkronisasikan.")
    
    if st.button("🔄 Sync & Update Database Vektor"):
        with st.spinner("Sedang memproses dokumen..."):
            try:
              response = requests.post(f"{BACKEND_URL}/api/sync")
              if response.status_code == 200:
                st.success(response.json()["message"])
              else:
                st.error(f"Gagal sinkronisasi: {response.json()['detail']}")
                
            except Exception as e:
              st.error(f"Tidak dapat terhubung ke backend API: {e}.")

# Inisialisasi Riwayat Chat di Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan chat yang sudah ada sebelumnya
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kolom Input Chat Pengguna
if user_query := st.chat_input("Tanyakan sesuatu tentang dokumenmu..."):
    # Tampilkan pesan user ke UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Ambil jawaban dari RAG Engine
    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            try:
              # tembak endpoint API
              response = requests.post(
                f"{BACKEND_URL}/api/chat",
                json={"input": user_query}
              )
              
              if response.status_code == 200:
                answer = response.json()["answer"]
                st.markdown(answer)
                
                # Simpan jawaban bot ke riwayat
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
              else:
                st.error(f"Error dari backend: {response.json()['detail']}")
                
            except Exception as e:
                st.error(f"Gagal terhubung dengan backend API. Error: {e}")