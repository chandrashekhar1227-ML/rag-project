"""
Layer 8: API Layer.
Wraps the existing pipeline (Layers 1-7) as a FastAPI service:
  POST /documents  - upload and ingest a PDF
  POST /query       - ask a question against whatever's been ingested
  GET  /health       - basic liveness check
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg2
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from chunk_by_section import get_full_text, chunk_by_section, SECTION_HEADERS
from setup_vector_db import create_table, insert_chunks, DB_CONFIG
from generate_answer import generate_answer

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Loaded once at startup, not per-request — re-loading a model on every
# call would make each request needlessly slow.
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    ml_models["embed_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    yield
    ml_models.clear()


app = FastAPI(title="Document Intelligence API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = UPLOAD_DIR / file.filename
    contents = await file.read()
    save_path.write_bytes(contents)

    text = get_full_text(str(save_path))
    try:
        chunks = chunk_by_section(text, SECTION_HEADERS)
    except ValueError as exc:
        # Our chunker only recognizes THIS document's exact headers (a
        # known Layer 2 limitation) - surface that clearly instead of a
        # confusing 500 error.
        raise HTTPException(
            status_code=422,
            detail=f"Could not chunk this document with our current section-based "
                   f"chunker: {exc}",
        )

    embeddings = ml_models["embed_model"].encode(chunks)

    conn = get_db_connection()
    try:
        create_table(conn)
        insert_chunks(conn, chunks, embeddings)
    finally:
        conn.close()

    return IngestResponse(filename=file.filename, chunks_ingested=len(chunks))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    conn = get_db_connection()
    try:
        answer_text = generate_answer(request.question, ml_models["embed_model"], conn)
    finally:
        conn.close()

    return QueryResponse(question=request.question, answer=answer_text)