#!/usr/bin/env bash
set -e

REPO="Openclaw"
DESCRIPTION="Production-grade AI agent examples: RAG pipeline, MCP server, LangChain ReAct agent, and data automation."

if ! command -v gh &>/dev/null; then
  echo "❌ 'gh' CLI not found. Install: https://cli.github.com/"
  exit 1
fi

if ! gh auth status &>/dev/null; then
  echo "❌ Not logged in. Run: gh auth login"
  exit 1
fi

gh repo create "$REPO" --description "$DESCRIPTION" --source=. --push --visibility public \
  && echo "✅ Repo created and pushed" \
  || {
    echo "ℹ️  Repo exists, pushing..."
    git remote set-url origin "https://github.com/$(gh api -q '.login')/$REPO.git"
    git push --set-upstream origin main
  }
