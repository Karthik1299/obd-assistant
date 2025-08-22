from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import os
from rag_pipeline import ingest_pdf, get_rag_chain

app = FastAPI()

# CORS for frontend
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# SQLite setup
DB_PATH = "../db/queries.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
conn.execute('''CREATE TABLE IF NOT EXISTS queries (id INTEGER PRIMARY KEY, code TEXT, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

class Query(BaseModel):
    code: str

@app.post("/ingest")
def ingest():
    ingest_pdf()
    return {"status": "Ingestion complete"}

@app.post("/query")
def query_obd(query: Query):
    chain = get_rag_chain()
    response = chain.run(query.code)
    conn.execute("INSERT INTO queries (code, response) VALUES (?, ?)", (query.code, response))
    conn.commit()
    return {"code": query.code, "response": response}

@app.get("/history")
def get_history(limit: int = 10):
    cursor = conn.execute("SELECT code, response, timestamp FROM queries ORDER BY timestamp DESC LIMIT ?", (limit,))
    history = [{"code": row[0], "response": row[1], "timestamp": row[2]} for row in cursor.fetchall()]
    return history

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)