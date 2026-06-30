"""
Pipeline — fetch + chunk + embed + store
Run: python pipeline/run.py
"""
import hashlib
import re
from datetime import datetime
from pathlib import Path

import requests
import feedparser
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb

from config import RSS_FEEDS, WEB_PAGES, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DB_DIR

# ── Chroma client ─────────────────────────────────────────────────────────────
DB_DIR.mkdir(exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
collection = chroma_client.get_or_create_collection(name="data_pipeline")

# ── Embedding model ────────────────────────────────────────────────────────────
model = SentenceTransformer(EMBEDDING_MODEL)

# ── Helpers ────────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def fetch_rss(url: str, tag: str):
    """Fetch RSS feed and return (title, content, url, source_tag) tuples."""
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:10]:  # limit to 10 items per feed
            content = entry.get("summary", entry.get("description", ""))
            clean = re.sub(r"<[^>]+>", "", content)  # strip HTML
            items.append((entry.title, clean, entry.link, tag))
        return items
    except Exception as e:
        print(f"Error fetching RSS {url}: {e}")
        return []

def fetch_web(url: str, tag: str):
    """Fetch a web page and return (title, content, url, source_tag) tuples."""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove script/style
        for elem in soup(["script", "style", "nav", "footer"]):
            elem.decompose()
        
        title = soup.find("title").string if soup.find("title") else url
        text = soup.get_text()
        text = re.sub(r"\s+", " ", text).strip()
        
        return [(title, text, url, tag)]
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

def run_pipeline():
    """Fetch → chunk → embed → store."""
    print("🚀 Starting pipeline...")
    
    docs = []
    
    # Fetch RSS feeds
    print("📡 Fetching RSS feeds...")
    for feed_config in RSS_FEEDS:
        docs.extend(fetch_rss(feed_config["url"], feed_config["tag"]))
    
    # Fetch web pages
    print("🌐 Fetching web pages...")
    for page_config in WEB_PAGES:
        docs.extend(fetch_web(page_config["url"], page_config["tag"]))
    
    print(f"📄 Collected {len(docs)} documents")
    
    # Chunk
    print("✂️  Chunking...")
    chunks_to_store = []
    for title, content, url, tag in docs:
        chunks = chunk_text(content)
        for chunk in chunks:
            chunks_to_store.append({
                "text": chunk,
                "metadata": {
                    "title": title[:100],
                    "url": url,
                    "source": tag,
                }
            })
    
    print(f"🔤 Created {len(chunks_to_store)} chunks")
    
    # Embed & store
    print("🔢 Embedding & storing...")
    embeddings = model.encode([c["text"] for c in chunks_to_store])
    
    for chunk, emb in zip(chunks_to_store, embeddings):
        chunk_id = hashlib.md5(chunk["text"].encode()).hexdigest()[:12]
        collection.add(
            ids=[chunk_id],
            embeddings=[emb.tolist()],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )
    
    print(f"✅ Stored {collection.count()} vectors")
    print(f"\nReady to query! Run: uvicorn pipeline.api:app --reload --port 8000")

if __name__ == "__main__":
    run_pipeline()
