"""
RAG Pipeline — LangChain + ChromaDB + OpenAI
Drop .pdf/.txt files into ./docs, then run.
"""
import os
import glob
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR   = Path(__file__).parent / "docs"
COLLECTION = "rag_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# ── 1. Load documents ─────────────────────────────────────────────────────────
def load_documents(directory: Path):
    docs = []
    for ext in ["*.pdf", "*.txt", "*.md"]:
        for file_path in glob.glob(str(directory / ext)):
            loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") \
                     else TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())
    if not docs:
        raise FileNotFoundError(
            f"No documents found in {directory}. Add .pdf, .txt, or .md files."
        )
    return docs

# ── 2. Split into chunks ──────────────────────────────────────────────────────
def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

# ── 3. Build vector store ─────────────────────────────────────────────────────
def get_vector_store(chunks, embeddings, collection_name=COLLECTION):
    return Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=collection_name,
        persist_directory=str(Path(__file__).parent / ".chroma_db"),
    )

# ── 4. Build RAG chain ────────────────────────────────────────────────────────
def build_rag_chain(vectorstore, llm_model=LLM_MODEL):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based ONLY on the provided context.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt | llm_model | StrOutputParser()
    )
    return chain

# ── 5. CLI ────────────────────────────────────────────────────────────────────
def main():
    docs = load_documents(DOCS_DIR)
    print(f"📄 Loaded {len(docs)} document(s)")
    chunks = split_documents(docs)
    print(f"✂️  {len(chunks)} chunks created")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vs = get_vector_store(chunks, embeddings)
    print(f"🔢 Indexed {vs._collection.count()} vectors")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = build_rag_chain(vs, llm)
    print("\n✅ Ready. Ask questions (Ctrl-C to exit).\n")
    while True:
        try:
            query = input("🔍 Question: ").strip()
            if not query:
                continue
            answer = chain.invoke(query)
            print(f"\n💬 Answer:\n{answer}\n" + "─" * 60)
        except KeyboardInterrupt:
            print("\n👋"); break

if __name__ == "__main__":
    main()
