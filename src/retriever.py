"""Retrieval with hybrid scoring + RBAC enforcement.

Flow:
    user, query
        |--> router.route()              -> list[str] of intents
        |--> rbac.chroma_filter(user)    -> pre-search metadata filter
        |--> vector_store.search()       -> candidate chunks (over-fetched)
        |--> intent re-ranking           -> bump chunks from routed silos
        |--> rbac.is_allowed() per chunk -> hard post-filter
        |--> top_k survivors             -> what the generator sees

The "over-fetch then filter" pattern is key: it means a chunk that was
correctly excluded by RBAC doesn't push a usable chunk out of the
top-k window.
"""

from __future__ import annotations

from src import config, router
from src.data_models import Chunk, RetrievedChunk, User
from src.rbac import RBACEngine
from src.vector_store import VectorStore


class Retriever:
    """Stateful retriever -- holds references to the vector store and the
    RBAC engine. One per pipeline."""

    def __init__(self, store: VectorStore, rbac: RBACEngine) -> None:
        self.store = store
        self.rbac = rbac

    # ------------------------------------------------------------------
    def retrieve(self, query: str, user: User) -> tuple[list[RetrievedChunk],
                                                         list[str]]:
        """Run the full retrieval pipeline for a (query, user) pair.

        Returns:
            (results, intents) where `results` is a list of RetrievedChunk
            in score order including BOTH allowed and denied chunks. The
            denied ones are kept so the audit trail / explainability layer
            can show "we found X but blocked it because Y."
        """
        intents = router.route(query)

        # Pre-filter the vector search by the cheap RBAC checks
        # (clearance + department). The expensive tag-based checks
        # happen post-search.
        where = self.rbac.chroma_filter(user)

        # Over-fetch: ask for more than we need so a few RBAC denials
        # don't starve the generator.
        raw = self.store.search(query, top_k=config.TOP_K, where=where)

        # ------------------------------------------------------------------
        # Score adjustment: small boost if the chunk's department matches
        # a routed intent. The router's intents include "employees" as a
        # virtual silo that maps to the HR department's salary records.
        # ------------------------------------------------------------------
        intent_to_dept = {
            "hr": "hr",
            "finance": "finance",
            "engineering": "engineering",
            "employees": "hr",        # employee directory lives in HR's silo
        }
        boosted: list[tuple[Chunk, float]] = []
        for chunk, score in raw:
            adj = score
            for intent in intents:
                if chunk.metadata.department == intent_to_dept.get(intent):
                    # +0.05 is enough to break ties without overriding
                    # genuine semantic similarity.
                    adj += 0.05
                    break
            boosted.append((chunk, adj))
        boosted.sort(key=lambda x: x[1], reverse=True)

        # ------------------------------------------------------------------
        # Final RBAC check on every chunk we plan to return. This is the
        # belt-and-braces step: if pre-filter logic ever drifts from the
        # full policy, this still catches the leak.
        # ------------------------------------------------------------------
        final: list[RetrievedChunk] = []
        for chunk, score in boosted:
            allowed, reason = self.rbac.is_allowed(user, chunk.metadata)
            final.append(RetrievedChunk(
                chunk=chunk,
                score=score,
                allowed=allowed,
                rbac_reason=reason,
            ))

        return final, intents
