"""ChromaDB persistence layer.

ChromaDB gives us:
    * Persistent storage on disk (rebuild once, reuse forever).
    * Native metadata filtering (`where` clauses) so RBAC pre-filtering
      happens inside the search call rather than being a post-step.
    * Cosine similarity out of the box.

We deliberately keep this wrapper thin -- the RAG-specific concepts
(chunks, metadata) live in `data_models.py`; here we just translate.
"""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings

from src import config, embeddings
from src.data_models import Chunk, DocumentMetadata


class VectorStore:
    """Wraps a single ChromaDB collection.

    We use a PersistentClient so the index survives process restarts. The
    collection name is fixed in config; if you want multiple isolated
    corpora you'd thread that through here.
    """

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(config.VECTOR_STORE_DIR),
            # Disable anonymized telemetry -- nice in an enterprise demo.
            settings=Settings(anonymized_telemetry=False),
        )
        # get_or_create avoids the "collection already exists" pain on
        # repeated runs of build_index.py. The distance metric matches
        # the embedding model: cosine for normalised vectors.
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop everything and start fresh. Used by build_index.py."""
        try:
            self.client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed + persist a batch of chunks.

        We do the embedding here (rather than letting Chroma do it via an
        embedding function) so the rest of the code can use the exact same
        embedding object for query-time similarity. That avoids the
        classic "your index uses model A but your queries use model B"
        debug nightmare.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vecs = embeddings.embed(texts)

        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=texts,
            embeddings=vecs.tolist(),
            metadatas=[c.metadata.to_chroma() | {"chunk_index": c.chunk_index}
                       for c in chunks],
        )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int,
               where: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]:
        """Run a similarity search, optionally pre-filtered by metadata.

        `where` is a ChromaDB filter expression (see rbac.RBACEngine.chroma_filter).

        Returns a list of (Chunk, score) pairs. We rehydrate the metadata
        dicts back into `DocumentMetadata` so downstream code stays typed.
        Score is converted from "distance" (0=identical, 2=opposite) to
        "similarity" (1=identical, -1=opposite) for readability.
        """
        qvec = embeddings.embed([query])[0]
        result = self.collection.query(
            query_embeddings=[qvec.tolist()],
            n_results=top_k,
            where=where,
        )

        # Chroma returns parallel lists. The shape is [[results_for_query_0]]
        # because it supports batched queries; we only ever send one query.
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]

        out: list[tuple[Chunk, float]] = []
        for cid, text, meta_dict, dist in zip(ids, docs, metas, distances):
            # tags were serialised as comma-separated -- split them back
            tags_str = meta_dict.get("tags", "") or ""
            md = DocumentMetadata(
                doc_id=meta_dict["doc_id"],
                source_path=meta_dict["source_path"],
                source_type=meta_dict["source_type"],
                department=meta_dict["department"],
                sensitivity=meta_dict["sensitivity"],
                title=meta_dict["title"],
                tags=[t for t in tags_str.split(",") if t],
            )
            chunk = Chunk(
                chunk_id=cid,
                text=text,
                metadata=md,
                chunk_index=meta_dict.get("chunk_index", 0),
            )
            # cosine distance -> cosine similarity
            similarity = 1.0 - float(dist)
            out.append((chunk, similarity))
        return out

    def count(self) -> int:
        """Mostly used in tests / health checks."""
        return self.collection.count()
