# Data Automation Pipeline

## Stack
- Fetch: `requests` + `BeautifulSoup` (web), `feedparser` (RSS)
- Embed: `sentence-transformers` (local, no API key)
- Store: `chromadb`
- API: `FastAPI`

## Flow
[Scheduler] → Fetch sources → Chunk → Embed → Store in Chroma → API serves queries

## Run
```bash
pip install requests beautifulsoup4 feedparser sentence-transformers chromadb fastapi uvicorn

# First run: fetch + embed all sources
python pipeline/run.py

# Start the query API
uvicorn pipeline.api:app --reload --port 8000

# Query:
curl "http://localhost:8000/query?q=your+question&top_k=5"
```
