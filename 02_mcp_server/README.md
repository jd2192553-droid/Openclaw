# MCP Server — Model Context Protocol Tools

## Stack
- Python stdlib + `requests`
- Minimal deps — no heavy framework

## Run
```bash
pip install requests
python mcp_server.py

# In your MCP client config:
# { "command": "python", "args": ["/path/to/mcp_server.py"] }
```

## Tools Exposed
- **web_search**: Brave Search API
- **fetch_url**: Extract text from URL
- **vectorstore_add**: Store text chunk
- **vectorstore_query**: Search local store
