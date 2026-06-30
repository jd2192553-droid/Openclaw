"""
FastAPI query API for the data pipeline.
Run: uvicorn pipeline.api:app --reload --port 8000
"""
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb

from config import EMBEDDING_MODEL, DB_DIR, TOP_K

# ── Chroma ─────────────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
collection = chroma_client.get_or_create_collection(name="data_pipeline")

# ── Embedding model (loaded once) ──────────────────────────────────────────────
model = SentenceTransformer(EMBEDDING_MODEL)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Data Pipeline Query API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryResponse(BaseModel):
    query: str
    results: list[dict]
    total_stored: int

@app.get("/query", response_model=QueryResponse)
def query(
    q: str = Query(..., description="Semantic search query"),
    top_k: int = Query(default=TOP_K, ge=1, le=50, description="Max results"),
    source: Optional[str] = Query(default=None, description="Filter by source tag"),
):
    """Semantic search over the indexed corpus."""
    embedding = model.encode([q]).tolist()[0]
    where_filter = {"source": source} if source else None

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":   doc,
            "source": meta.get("source", ""),
            "url":    meta.get("url", ""),
            "title":  meta.get("title", ""),
            "kind":   meta.get("kind", ""),
            "score":  round(1 - dist, 4),
        })

    return QueryResponse(query=q, results=hits, total_stored=collection.count())

@app.get("/stats")
def stats():
    """Return total count and source breakdown."""
    all_data = collection.get(include=["metadatas"])
    sources = {}
    for meta in all_data["metadatas"]:
        tag = meta.get("source", "unknown")
        sources[tag] = sources.get(tag, 0) + 1
    return {"total_vectors": collection.count(), "by_source": sources}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
