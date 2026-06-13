"""Query router.

We don't strictly *need* a router -- the vector store could handle every
query by similarity alone. But routing earns its keep in two ways:

    1. It biases retrieval toward the right silo. A question about
       "parental leave" should pull from HR, not coincidentally-worded
       finance text.
    2. It gives the explainability output a human-friendly label
       ("routed to: hr, finance") that helps users understand the answer.

The implementation is intentionally simple: a keyword classifier first
(fast, deterministic, easy to audit) and embedding-similarity fall-back
for queries the keywords miss.
"""

from __future__ import annotations

from src import embeddings


# Keyword -> intent. Order doesn't matter; a query can match multiple
# intents and we return all matches so the retriever can search broadly.
INTENT_KEYWORDS: dict[str, list[str]] = {
    "hr": [
        "leave", "parental", "maternity", "paternity", "handbook",
        "remote work", "code of conduct", "performance review",
        "vacation", "benefits", "harassment",
    ],
    "finance": [
        "revenue", "earnings", "profit", "margin", "forecast", "q1", "q2",
        "q3", "q4", "financial", "budget", "opex", "capex", "guidance",
    ],
    "engineering": [
        "security", "audit", "incident", "iam", "vulnerability",
        "deployment", "infrastructure", "ssh", "bastion", "ransomware",
        "logs", "alert",
    ],
    "employees": [
        "salary", "compensation", "pay", "wage", "hire date", "tenure",
        "employee", "directory",
    ],
}


# Canonical "exemplar" phrases per intent for the embedding fallback. These
# give the embedding-based classifier a stable anchor to compare against.
INTENT_EXEMPLARS: dict[str, str] = {
    "hr": "human resources, employee handbook, leave policy, benefits, "
          "code of conduct, remote work",
    "finance": "quarterly financial report, revenue, earnings, operating "
               "expenses, forecast, profit margin",
    "engineering": "security audit, infrastructure incident, IAM permissions, "
                   "vulnerability, deployment, system logs",
    "employees": "employee directory, salaries, compensation, hire dates, "
                 "tenure information",
}


# Pre-computed once; module-level lazy init lives in `_exemplar_vectors`.
_EXEMPLAR_CACHE: dict[str, list[float]] | None = None


def _exemplar_vectors() -> dict[str, list[float]]:
    """Embed the intent exemplars once and cache them.

    We avoid embedding at import time because that triggers model
    download/load even for callers who only want keyword routing.
    """
    global _EXEMPLAR_CACHE
    if _EXEMPLAR_CACHE is None:
        names = list(INTENT_EXEMPLARS.keys())
        texts = [INTENT_EXEMPLARS[n] for n in names]
        vecs = embeddings.embed(texts)
        _EXEMPLAR_CACHE = {n: v.tolist() for n, v in zip(names, vecs)}
    return _EXEMPLAR_CACHE


def _keyword_intents(query: str) -> list[str]:
    """Return all intents whose keywords appear in the query.

    Substring matching is fine for short queries. We lowercase both sides
    so "Q4 Revenue" matches the "q4" / "revenue" keywords.
    """
    q = query.lower()
    hits: list[str] = []
    for intent, words in INTENT_KEYWORDS.items():
        if any(w in q for w in words):
            hits.append(intent)
    return hits


def _embedding_intent(query: str) -> str:
    """Cosine-similarity nearest exemplar.

    Used as the fallback when no keyword matches. Always returns one
    intent; if the query truly matches none of our buckets we'll still
    return the closest, which is the right behaviour for a router (it's
    the *retriever's* job to decide whether the result is actually
    relevant).
    """
    import numpy as np

    qv = embeddings.embed([query])[0]
    exemplars = _exemplar_vectors()
    best, best_score = next(iter(exemplars)), -1.0
    for intent, ev in exemplars.items():
        # vectors are L2-normalised so dot == cosine similarity
        score = float(np.dot(qv, np.asarray(ev, dtype="float32")))
        if score > best_score:
            best, best_score = intent, score
    return best


def route(query: str) -> list[str]:
    """Public entry point. Returns one or more intents for the query.

    The returned list is used by the retriever to weight or restrict the
    search. An empty list would mean "no signal -- search globally", but
    we always return at least one intent (via the embedding fallback).
    """
    hits = _keyword_intents(query)
    if hits:
        return hits
    return [_embedding_intent(query)]
