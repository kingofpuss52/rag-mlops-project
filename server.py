from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import ingest_documents, get_rag_chain

app = FastAPI(title="RAG MLOps Backend API")

# Model data untuk request chat
class ChatRequest(BaseModel):
    input: str

@app.get("/")
def read_root():
    return {"status": "Backend API is running"}

@app.post("/api/sync")
def sync_database():
    try:
        status_msg = ingest_documents()
        return {"status": "success", "message": status_msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_with_rag(request: ChatRequest):
    try:
        rag_chain = get_rag_chain()
        answer = rag_chain.invoke(request.input)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))