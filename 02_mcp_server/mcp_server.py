"""
MCP Server — Model Context Protocol (JSON-RPC 2.0 over stdio)
Provides: web_search, fetch_url, vectorstore_add, vectorstore_query
"""
import json, sys, os, re
from typing import Optional
import requests

TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": "Search the web using Brave Search API.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Extract readable text from a URL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "default": 4000},
            },
            "required": ["url"],
        },
    },
    {
        "name": "vectorstore_add",
        "description": "Store a text chunk in the local vector store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "metadata": {"type": "object", "default": {}},
            },
            "required": ["text"],
        },
    },
    {
        "name": "vectorstore_query",
        "description": "Search the local vector store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
]

# ── In-memory vector store (keyword scoring — swap for real embeddings) ────────
class SimpleVectorStore:
    def __init__(self):
        self._texts, self._metadata, self._count = [], [], 0

    def add(self, text: str, metadata: Optional[dict] = None) -> dict:
        idx = self._count
        self._texts.append(text)
        self._metadata.append(metadata or {"id": idx})
        self._count += 1
        return {"id": str(idx), "stored": True}

    def query(self, query: str, top_k: int = 5) -> list[dict]:
        q_words = set(re.findall(r"\w+", query.lower()))
        scores = []
        for i, text in enumerate(self._texts):
            t_words = set(re.findall(r"\w+", text.lower()))
            score = len(q_words & t_words)
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [{"id": str(i), "text": self._texts[i],
                 "metadata": self._metadata[i], "score": s}
                for s, i in scores[:top_k]]

VS = SimpleVectorStore()

# ── Tool implementations ──────────────────────────────────────────────────────
def tool_web_search(query: str, count: int = 5) -> dict:
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return {"error": "BRAVE_SEARCH_API_KEY not set"}
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        params={"q": query, "count": min(count, 10)}, timeout=10,
    )
    if not resp.ok:
        return {"error": f"Brave API error {resp.status_code}"}
    results = []
    for item in resp.json().get("web", {}).get("results", [])[:count]:
        results.append({"title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("description", "")})
    return {"query": query, "results": results}

def tool_fetch_url(url: str, max_chars: int = 4000) -> dict:
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"Failed to fetch {url}: {e}"}
    text = resp.text
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>|<nav.*?</nav>|<footer.*?</footer>", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return {"url": url, "content": text[:max_chars], "truncated": len(text) > max_chars}

def tool_vectorstore_add(text: str, metadata: Optional[dict] = None) -> dict:
    return VS.add(text, metadata)

def tool_vectorstore_query(query: str, top_k: int = 5) -> dict:
    return {"query": query, "results": VS.query(query, top_k)}

TOOL_IMPLS = {
    "web_search": tool_web_search,
    "fetch_url": tool_fetch_url,
    "vectorstore_add": tool_vectorstore_add,
    "vectorstore_query": tool_vectorstore_query,
}

# ── JSON-RPC 2.0 over stdio ────────────────────────────────────────────────────
def read_message():
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "":
            break
        lines.append(line.strip())
    return json.loads(" ".join(lines)) if lines else None

def send_response(req_id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
    sys.stdout.flush()

def send_error(req_id, code, message):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id,
                                  "error": {"code": code, "message": message}}) + "\n")
    sys.stdout.flush()

def handle_message(msg: dict):
    method, req_id = msg.get("method", ""), msg.get("id")
    if method == "tools/list":
        send_response(req_id, {"tools": TOOL_DEFINITIONS})
    elif method == "tools/call":
        params = msg.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if tool_name not in TOOL_IMPLS:
            send_error(req_id, -32601, f"Unknown tool: {tool_name}")
            return
        try:
            result = TOOL_IMPLS[tool_name](**arguments)
            send_response(req_id, {"content": [{"type": "text", "text": json.dumps(result)}]})
        except Exception as e:
            send_error(req_id, -32603, f"Tool error: {e}")
    elif method == "ping":
        send_response(req_id, {"pong": True})
    else:
        send_error(req_id, -32601, f"Unknown method: {method}")

def main():
    sys.stdout.write(json.dumps({
        "jsonrpc": "2.0", "id": None,
        "result": {"protocolVersion": "2024-11-05",
                   "capabilities": {"tools": {}},
                   "serverInfo": {"name": "example-mcp-server", "version": "1.0.0"}},
    }) + "\n")
    sys.stdout.flush()
    while True:
        msg = read_message()
        if msg is None:
            break
        handle_message(msg)

if __name__ == "__main__":
    main()
