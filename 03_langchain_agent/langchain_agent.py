"""
LangChain Agent — ReAct loop
Tools: web search, Wikipedia, calculator, persistent memory
"""
import sqlite3, os
from datetime import datetime
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

DB_PATH = os.environ.get("AGENT_MEMORY_DB",
                         str(os.path.expanduser("~/.agent_memory.db")))
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# ── Memory (SQLite) ────────────────────────────────────────────────────────────
def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fact TEXT NOT NULL, source TEXT, created_at TEXT NOT NULL)""")
    conn.commit()
    conn.close()
_init_db()

@tool
def memory_search(query: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT fact, source, created_at FROM facts WHERE fact LIKE ? ORDER BY id DESC LIMIT 10",
        (f"%{query}%",))
    rows = cur.fetchall()
    conn.close()
    return "\n".join(f"[{src} @ {ts}] {fact}" for fact, src, ts in rows) or "No matches."

@tool
def memory_save(fact: str, source: Optional[str] = None) -> str:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO facts (fact, source, created_at) VALUES (?, ?, ?)",
                 (fact, source or "user", datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return f"Saved: {fact}"

# ── Calculator (safe eval) ────────────────────────────────────────────────────
@tool
def calculate(expression: str) -> str:
    import math
    allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
               "log": math.log, "pi": math.pi, "e": math.e, "floor": math.floor,
               "ceil": math.ceil, "abs": abs, "pow": pow, "log10": math.log10, "exp": math.exp}
    safe = {"__builtins__": {}, "math": allowed}
    try:
        result = eval(expression, safe, allowed)
        return str(result)
    except Exception as ex:
        return f"Error: {ex}"

# ── Web search ────────────────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return "Error: BRAVE_SEARCH_API_KEY not set."
    import requests
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            params={"q": query, "count": 3}, timeout=10)
        out = []
        for item in resp.json().get("web", {}).get("results", [])[:3]:
            out.append(f"- {item.get('title')}: {item.get('url')}\n  {item.get('description')}")
        return "\n\n".join(out) if out else "No results."
    except Exception as e:
        return f"Search error: {e}"

# ── Wikipedia ─────────────────────────────────────────────────────────────────
@tool
def wikipedia(query: str) -> str:
    try:
        import wikipedia as wiki
        wiki.set_lang("en")
        return wiki.summary(query, sentences=3)
    except Exception as ex:
        return f"Wikipedia error: {ex}"

# ── Agent ────────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
tools = [memory_search, memory_save, calculate, web_search, wikipedia]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant with access to search, Wikipedia, math, and a persistent memory database."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

if __name__ == "__main__":
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        result = executor.invoke({"input": user_input, "chat_history": chat_history})
        print(f"Agent: {result['output']}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=result["output"]))
