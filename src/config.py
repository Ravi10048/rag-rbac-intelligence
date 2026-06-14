"""Central configuration.

Everything that another module might want to tweak lives here so we don't have
magic numbers scattered around. Anything that is sensitive (API keys) is read
from the environment, not hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

# Try to load a local .env if python-dotenv is installed. We do this *softly*
# so that the package still imports cleanly even without the optional dep.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
# Resolved relative to the project root (parent of the `src/` folder). Using
# absolute paths everywhere avoids "cwd surprises" when the pipeline is run
# from a notebook vs. from the CLI.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"
DB_DIR = DATA_DIR / "databases"
LOGS_DIR = DATA_DIR / "logs"
POLICIES_DIR = DATA_DIR / "policies"

# ChromaDB persists to disk so we can rebuild the index once and reuse it.
VECTOR_STORE_DIR = PROJECT_ROOT / ".chroma"
COLLECTION_NAME = "enterprise_rag"


# ---------------------------------------------------------------------------
# Embedding + chunking knobs
# ---------------------------------------------------------------------------
# all-MiniLM-L6-v2 is a sensible default: small (~80MB), fast on CPU, and good
# enough for semantic search on English enterprise text. It produces 384-dim
# vectors so the vector store stays cheap.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunk size is a tradeoff: too small and we lose context, too large and the
# embedding gets diluted. 600 chars works well for narrative PDFs while still
# fitting structured rows (CSV rows are usually < 200 chars anyway).
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# Top-K candidates BEFORE the RBAC post-filter. We retrieve a few extra so
# that even if some are dropped for permission reasons we still have enough
# context for the generator.
TOP_K = int(os.getenv("TOP_K", "8"))
FINAL_K = int(os.getenv("FINAL_K", "4"))


# ---------------------------------------------------------------------------
# Sensitivity levels
# ---------------------------------------------------------------------------
# Ordered low -> high. The RBAC engine compares numerically so adding a new
# level just means inserting it here.
SENSITIVITY_LEVELS = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}


# ---------------------------------------------------------------------------
# LLM backend discovery
# ---------------------------------------------------------------------------
# The generator probes these in order at startup. The first one that has a
# usable API key / reachable server wins. If none do, the pipeline falls back
# to a deterministic extractive answer (see generator.FallbackBackend).
#
# Groq is the preferred backend when a key is set: a free, very fast, hosted
# OpenAI-compatible API — no local model to install (unlike Ollama) and it works
# in cloud deploys too. Get a free key at https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# llama3.2:1b is small (~1.3GB), fast on CPU, and good enough for grounded
# summarisation tasks. Bigger models (llama3.2:3b, mistral, etc.) give nicer
# prose at the cost of latency -- override via env var if desired.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
