# LangChain Agent — ReAct (Reason + Act Loop)

## Stack
- LangChain (OpenAI Functions agent, tool calling)
- Tools: Brave Search, Wikipedia, Calculator, Memory (SQLite)

## Run
```bash
pip install langchain langchain-openai brave-search wikipedia
export OPENAI_API_KEY=...
export BRAVE_SEARCH_API_KEY=...
python langchain_agent.py
```

## Features
- **Web Search**: Brave Search API integration
- **Wikipedia**: Quick knowledge lookups
- **Calculator**: Safe math evaluation
- **Memory**: SQLite-backed fact storage & retrieval
