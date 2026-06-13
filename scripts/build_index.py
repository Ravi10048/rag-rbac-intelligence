"""Build (or rebuild) the ChromaDB index from the synthetic dataset.

Run once after `generate_data.py`:

    python scripts/build_index.py

This is idempotent -- it drops the existing collection and re-ingests.
The first run downloads the embedding model (~80 MB), subsequent runs
are fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the script runnable directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ingestion
from src.vector_store import VectorStore


def main() -> None:
    print("Loading + chunking source documents...")
    chunks = ingestion.load_all()
    if not chunks:
        print("No chunks produced. Did you run scripts/generate_data.py?")
        sys.exit(1)
    print(f"  -> {len(chunks)} chunks ready for embedding")

    print("Embedding + writing to ChromaDB (this is slow on first run)...")
    store = VectorStore()
    store.reset()
    store.add_chunks(chunks)

    print(f"Done. Collection now holds {store.count()} chunks.")


if __name__ == "__main__":
    main()
