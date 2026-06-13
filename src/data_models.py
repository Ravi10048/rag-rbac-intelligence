"""Typed containers shared across the pipeline.

These are intentionally lightweight dataclasses rather than Pydantic models -
the project doesn't need schema validation at the API boundary (it has no
public API), and dataclasses keep the dependency surface small.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Identity & policy
# ---------------------------------------------------------------------------
@dataclass
class User:
    """Represents an employee querying the system.

    `clearance` mirrors the sensitivity levels in config.SENSITIVITY_LEVELS -
    the RBAC engine compares the user's clearance against each document's
    sensitivity to decide read access.
    """

    user_id: str
    name: str
    department: str          # e.g. "engineering", "hr", "finance"
    role: str                # e.g. "manager", "analyst", "ic"
    clearance: str           # one of SENSITIVITY_LEVELS keys

    # Departments the user is allowed to query into. A "cross-functional"
    # role like CEO/CISO can have multiple departments here.
    accessible_departments: list[str] = field(default_factory=list)


@dataclass
class AccessPolicy:
    """A single rule in the access policy file.

    The engine evaluates the *most specific* matching rule first - see
    rbac.RBACEngine.is_allowed for the resolution order.
    """

    name: str
    description: str
    allowed_roles: list[str] = field(default_factory=list)         # role names
    allowed_departments: list[str] = field(default_factory=list)   # dept names
    min_clearance: str = "public"


# ---------------------------------------------------------------------------
# Documents & chunks
# ---------------------------------------------------------------------------
@dataclass
class DocumentMetadata:
    """Metadata attached to every chunk so the retriever can RBAC-filter
    *before* the vector search runs (cheap) and re-check *after* (defense
    in depth)."""

    doc_id: str
    source_path: str
    source_type: str          # "pdf" | "csv" | "json"
    department: str           # owning department
    sensitivity: str          # "public" | "internal" | "confidential" | "restricted"
    title: str
    tags: list[str] = field(default_factory=list)

    def to_chroma(self) -> dict[str, Any]:
        """ChromaDB only stores scalar metadata values, so list-valued
        fields (`tags`) get joined into a comma-separated string. The
        retriever splits them back out when displaying."""
        d = asdict(self)
        d["tags"] = ",".join(self.tags)
        return d


@dataclass
class Chunk:
    """A single retrievable unit of text."""

    chunk_id: str             # stable id we can cite back to
    text: str
    metadata: DocumentMetadata
    chunk_index: int          # position within the source document


# ---------------------------------------------------------------------------
# Pipeline outputs
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    """A chunk returned from search, with its similarity score and the
    RBAC decision that was applied to it (handy for the audit trail)."""

    chunk: Chunk
    score: float              # cosine similarity (1.0 = identical)
    allowed: bool             # final RBAC decision
    rbac_reason: str          # human-readable explanation


@dataclass
class PipelineTrace:
    """Step-by-step record of what the pipeline did for one query.

    Where the per-chunk audit trail (`RAGResponse.retrieved`) answers
    *which documents were considered and what was the decision on each*,
    this trace answers *what happened at every stage of the pipeline* -
    identity resolution, routing, the actual ChromaDB filter clause,
    which LLM was called, total latency, etc.

    Together they give a compliance/security officer the complete story:
    "user X asked Y at time T, we identified them as role R, routed to
    silos S, applied filter F at the vector store, retrieved N chunks of
    which K were blocked because P, sent M chunks to model Z, returned
    answer in L ms."
    """

    # --- Stage 1: Identity resolution -------------------------------------
    user_email: str
    user_id: str
    user_role: str
    user_department: str
    user_clearance: str
    accessible_departments: list[str]

    # --- Stage 2: Query routing -------------------------------------------
    detected_intents: list[str]

    # --- Stage 3: Pre-retrieval RBAC filter -------------------------------
    # The exact ChromaDB `where` clause that gated the vector search.
    # `None` means the user was a wildcard (executive) so no filter applied.
    chroma_where_clause: dict | None

    # --- Stage 4: Retrieval ------------------------------------------------
    embedding_model: str
    top_k_requested: int
    candidates_retrieved: int

    # --- Stage 5: Post-retrieval RBAC --------------------------------------
    chunks_allowed: int
    chunks_denied: int
    denied_reasons_summary: list[str]   # deduped reason strings

    # --- Stage 6: Generation ----------------------------------------------
    llm_backend: str          # "ollama" / "openai" / "anthropic" / "fallback-extractive"
    llm_model: str            # the actual model identifier
    context_chunks_to_llm: int
    refused_before_llm: bool  # True if RBAC blocked everything (LLM never called)

    # --- Stage 7: Response timing -----------------------------------------
    timestamp_iso: str        # human-readable ISO timestamp
    elapsed_ms: float         # total wall-clock for the .ask() call


@dataclass
class RAGResponse:
    """End-to-end response returned by the pipeline.

    Fields are designed so the demo / UI can show *why* the system answered
    the way it did - this is the "explainability" goal of the system.
    """

    query: str
    user: User
    answer: str
    citations: list[str]           # human-readable citations
    retrieved: list[RetrievedChunk]  # full audit trail (allowed + denied)
    routed_to: list[str]           # which silos/intents we hit
    confidence: float              # 0.0 - 1.0
    refused: bool = False          # True if RBAC blocked the entire answer
    refusal_reason: str = ""
    # Populated by the pipeline orchestrator AFTER generation completes.
    # Optional so older callers that build RAGResponse directly still work.
    trace: PipelineTrace | None = None
