import streamlit as st
import requests

# ==========================================
# Konfigurasi Halaman Web
# ==========================================
st.set_page_config(
    page_title="RAG SOP Tekstil", 
    page_icon="🧵", 
    layout="centered"
)

st.title("🤖 AI Assistant - SOP Deteksi Cacat Tekstil")
st.caption("Sistem tanya jawab internal dengan Guardrails NLP dan RAG Pipeline.")
st.divider()

# ==========================================
# Manajemen Memori Obrolan (Session State)
# ==========================================
# Agar teks percakapan tidak hilang saat halaman dimuat ulang
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan seluruh riwayat obrolan di layar
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# Kolom Input Chat
# ==========================================
if prompt := st.chat_input("Uji sistem ini atau tanyakan seputar SOP mesin tekstil..."):
    
    # 1. Tampilkan pesan user di layar & simpan ke memori
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Siapkan area untuk jawaban AI
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("*⏳ Sistem sedang memeriksa keamanan dan mencari dokumen...*")
        
        try:
            # 3. Tembak data ke FastAPI Backend kita
            # Pastikan uvicorn api:app sedang menyala di terminal terpisah!
            history_untuk_api = st.session_state.messages[:-1]
            
            response = requests.post(
                "http://127.0.0.1:8000/api/v1/query",
                json={
                    "pertanyaan": prompt,
                    "riwayat_chat": history_untuk_api
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                jawaban_ai = data["jawaban"]
                waktu_eksekusi = data["waktu_eksekusi_detik"]
                sumber = data["sumber_dokumen"]
                
                # Tampilkan jawaban
                placeholder.markdown(jawaban_ai)
                
                # Tampilkan metrik kecepatan di bawah jawaban
                st.caption(f"⏱️ **Waktu Inferensi:** {waktu_eksekusi} detik")
                
                # (Opsional) Jika ingin melihat teks asli yang diambil ChromaDB
                with st.expander("Lihat Referensi Dokumen (Chunks)"):
                    for i, doc in enumerate(sumber):
                        st.info(f"**Chunk {i+1}:**\n{doc}")
                
                # Simpan jawaban AI ke memori percakapan
                st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})
                
            else:
                placeholder.error(f"⚠️ Error dari Server: {response.text}")
                
        except requests.exceptions.ConnectionError:
            placeholder.error("🚨 Gagal terhubung ke Backend! Pastikan server FastAPI (api.py) sudah berjalan.")