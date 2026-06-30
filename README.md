# Examples — AI Agent Stack

Four production-grade examples for the Python AI agent stack.

```
examples/
├── 01_rag_pipeline/          LangChain + Chroma + PDF/TXT → Q&A
├── 02_mcp_server/            MCP server (JSON-RPC 2.0 stdio) — tools over protocol
├── 03_langchain_agent/       ReAct agent: search + Wikipedia + calculator + memory
└── 04_data_automation/       Fetch → chunk → embed → Chroma → FastAPI query
```

## Quick start

### 1 — RAG Pipeline
```bash
cd 01_rag_pipeline
pip install langchain langchain-community langchain-openai chromadb openai tiktoken
# drop .pdf/.txt files into ./docs
python rag_pipeline.py
```

### 2 — MCP Server
```bash
cd 02_mcp_server
pip install requests
python mcp_server.py  # runs on stdio, connect via MCP client
```

### 3 — LangChain Agent
```bash
cd 03_langchain_agent
pip install langchain langchain-openai brave-search wikipedia
export OPENAI_API_KEY=...
export BRAVE_SEARCH_API_KEY=...
python langchain_agent.py
```

### 4 — Data Automation
```bash
cd 04_data_automation
pip install requests beautifulsoup4 feedparser sentence-transformers chromadb fastapi uvicorn
python run.py          # fetch + embed + store
uvicorn pipeline.api:app --port 8000  # query API
curl "http://localhost:8000/query?q=your+question&top_k=5"
```
