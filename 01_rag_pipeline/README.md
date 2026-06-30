# RAG Pipeline — LangChain + Chroma + PDF

## Stack
- LangChain (document loaders, text splitting, embeddings, chain)
- ChromaDB (vector store)
- OpenAI (or Ollama for local)

## Run
```bash
pip install langchain langchain-community chromadb openai tiktoken
export OPENAI_API_KEY=...
python rag_pipeline.py
```
