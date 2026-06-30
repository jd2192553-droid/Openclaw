"""
Data Automation Pipeline — fetch → chunk → embed → store → query API
"""
from pathlib import Path
from pydantic import BaseModel

# ── Sources to crawl ───────────────────────────────────────────────────────────
class Source(BaseModel):
    url: str
    kind: str  # "rss" | "web" | "csv"
    tag: str   # label for filtering (e.g. "tech_news", "finance")

RSS_FEEDS = [
    {"url": "https://hnrss.org/frontpage",           "kind": "rss", "tag": "hackernews"},
    {"url": "https://feeds.bbci.co.uk/news/rss.xml", "kind": "rss", "tag": "bbc_news"},
]

WEB_PAGES = [
    {"url": "https://en.wikipedia.org/wiki/Main_Page", "kind": "web", "tag": "wikipedia"},
]

# ── Processing config ──────────────────────────────────────────────────────────
CHUNK_SIZE        = 300   # characters per chunk
CHUNK_OVERLAP     = 50    # overlap between chunks
TOP_K             = 5     # default results to return
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"  # local, no API key
DB_DIR            = Path(__file__).parent / ".chroma_db"
