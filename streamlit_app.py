"""Streamlit UI for the Enterprise RAG demo.

Run with:
    streamlit run streamlit_app.py

The UI is a thin presentation layer on top of `src.rag_pipeline.RAGPipeline`.
It exists to make the RBAC story visible -- the same query asked by different
personas produces visibly different responses, with the allowed/denied audit
trail rendered side by side.

We intentionally keep the styling minimal so reviewers can see the data
clearly rather than admiring the chrome.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when streamlit is invoked from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.data_models import RAGResponse
from src.rag_pipeline import RAGPipeline
from src.rbac import RBACEngine


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise RAG - RBAC Demo",
    layout="wide",
    page_icon=":lock:",
)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
# Building the pipeline loads the embedding model + opens ChromaDB. That's
# ~10s on cold start, so we cache it across Streamlit reruns.
@st.cache_resource(show_spinner="Loading RAG pipeline (one-time cold start)...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource
def get_rbac() -> RBACEngine:
    # Used independently for the user dropdown so we don't have to spin up
    # the whole pipeline to populate the sidebar.
    return RBACEngine()


# ---------------------------------------------------------------------------
# Sidebar - persona switcher
# ---------------------------------------------------------------------------
rbac = get_rbac()
all_users = rbac.list_users()
email_to_user = {u.user_id: u for u in all_users}

# We index personas by email rather than user_id because the user_id field
# is internal -- email is what a real auth system would surface.
sidebar_options = {
    "alice@acme.com": "Alice Chen  -  HR Manager",
    "bob@acme.com": "Bob Singh  -  Software Engineer",
    "carol@acme.com": "Carol Diaz  -  Finance Analyst",
    "david@acme.com": "David Patel  -  CEO (full access)",
}

with st.sidebar:
    st.title(":lock: Acme Corp RAG")
    st.caption("Enterprise assistant with strict RBAC")

    selected_email = st.radio(
        "Sign in as:",
        options=list(sidebar_options.keys()),
        format_func=lambda e: sidebar_options[e],
        index=0,
    )
    user = rbac.get_user(selected_email)

    st.divider()
    st.subheader("Your profile")
    st.markdown(f"""
    * **Name:** {user.name}
    * **Role:** `{user.role}`
    * **Department:** `{user.department}`
    * **Clearance:** `{user.clearance}`
    * **Can read silos:** `{', '.join(user.accessible_departments) or '* (all)'}`
    """)

    st.divider()
    st.subheader("Try these queries")
    sample_queries = [
        "What is the parental leave policy?",
        "What is Bob Singh's salary?",
        "What were our Q4 2025 revenue numbers?",
        "Tell me about the payment-staging credit card exposure incident.",
        "Summarise Q4 revenue and any major security findings.",
    ]
    # Clicking a sample button does two things on the NEXT rerun:
    #   1. Fills the main query box (via session_state["query_text"]).
    #   2. Auto-submits the query (via session_state["auto_submit"]).
    # This avoids the "click sample then nothing happens" UX trap.
    for q in sample_queries:
        if st.button(q, use_container_width=True, key=f"sample_{hash(q)}"):
            st.session_state["query_text"] = q
            st.session_state["auto_submit"] = True
            st.rerun()


# ---------------------------------------------------------------------------
# Main panel - query + response
# ---------------------------------------------------------------------------
st.title("Ask the enterprise assistant")
st.caption(
    "The same question may yield different answers depending on who is asking. "
    "Each retrieved chunk is independently access-checked; chunks you can't "
    "see are blocked and the reason is logged."
)

# We bind the text input to session state via `key=` so sidebar sample-query
# buttons can programmatically update its value. Without the key the widget
# would only honour `value=` on the first render.
if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""

query = st.text_input(
    "Your question",
    key="query_text",
    placeholder="e.g. What is the parental leave policy?",
    label_visibility="collapsed",
)

# Ask button is always enabled. If empty we show a friendly warning instead
# of silently failing -- much less confusing than a disabled grey button.
ask = st.button("Ask", type="primary")

# Auto-submit flag set by sidebar sample-query buttons.
auto_submit = st.session_state.pop("auto_submit", False)


def render_response(resp: RAGResponse) -> None:
    """Render a single RAGResponse with all explainability surfaces."""

    # Top status row - routing + refusal + confidence.
    #
    # The confidence metric has two distinct semantics:
    #   - When answered: "retrieval confidence" = mean similarity of the
    #     chunks shown to the LLM. Higher = corpus had good matches.
    #   - When refused:  showing 0.00 would be misleading because the
    #     refusal decision is *deterministic* (rule-based), not statistical.
    #     We display 100% with a "Policy match" label instead.
    cols = st.columns([2, 2, 2, 2])
    cols[0].metric("Asked as", resp.user.name)
    cols[1].metric("Routed to", ", ".join(resp.routed_to) or "-")
    cols[2].metric(
        "Status",
        "REFUSED" if resp.refused else "Answered",
        delta=resp.refusal_reason if resp.refused else None,
        delta_color="inverse" if resp.refused else "normal",
    )
    if resp.refused:
        cols[3].metric(
            "Policy match",
            "100%",
            delta="deterministic",
            delta_color="off",
        )
    else:
        cols[3].metric("Confidence", f"{resp.confidence:.2f}")

    # Status bar under the metrics.
    if resp.refused:
        # Solid red bar = full certainty on the refusal decision.
        st.progress(1.0, text="RBAC decision: deterministic (rule-based, "
                              "no statistical uncertainty)")
    else:
        bar = min(max(resp.confidence, 0.0), 1.0)
        st.progress(bar, text=f"Retrieval confidence: {bar:.0%}")

    st.divider()

    # ANSWER - the big readable block
    st.subheader("Answer")
    if resp.refused:
        # Red callout for refusal so it's instantly visible.
        st.error(resp.answer)
    else:
        st.markdown(resp.answer)

    # CITATIONS - clickable would require a real file server; we just list.
    if resp.citations:
        st.subheader("Citations")
        for c in resp.citations:
            st.markdown(f"* {c}")

    # AUDIT TRAIL - the explainability money shot. Expanded by default so
    # the per-chunk RBAC decisions are immediately visible without an extra
    # click; collapsing is one click if the reader doesn't care.
    n_allowed = sum(r.allowed for r in resp.retrieved)
    n_total = len(resp.retrieved)
    st.subheader("Retrieval traceability (audit trail)")
    with st.expander(
        f"📋 Showing {n_allowed} of {n_total} retrieved chunks "
        f"({n_total - n_allowed} blocked by RBAC) - click to collapse",
        expanded=True,
    ):
        st.caption(
            "Every candidate chunk the retriever surfaced, with its similarity "
            "score and the access-control decision the RBAC engine applied. "
            "Denied chunks were never shown to the language model."
        )
        for r in resp.retrieved:
            badge = ":green[ALLOW]" if r.allowed else ":red[DENY]"
            sens_colour = {
                "public":       ":blue[public]",
                "internal":     ":blue[internal]",
                "confidential": ":orange[confidential]",
                "restricted":   ":red[restricted]",
            }.get(r.chunk.metadata.sensitivity, r.chunk.metadata.sensitivity)

            st.markdown(
                f"{badge}  &nbsp;  score=`{r.score:+.2f}`  &nbsp;  "
                f"`{r.chunk.chunk_id}`  &nbsp;  "
                f"({r.chunk.metadata.source_type}, sensitivity={sens_colour})"
            )
            st.caption(f"&nbsp;&nbsp;&nbsp;reason: {r.rbac_reason}")
            # Show the chunk text but truncated -- full text is in citations.
            preview = r.chunk.text[:240] + ("..." if len(r.chunk.text) > 240 else "")
            st.code(preview, language=None)

    # ------------------------------------------------------------------
    # FULL PIPELINE TRACE - end-to-end audit record
    # ------------------------------------------------------------------
    # The chunk-level audit trail above answers "which docs and why";
    # this section answers "what did the whole pipeline DO" - identity,
    # routing, the actual filter clause sent to the vector store, which
    # LLM was used, total latency. This is the record an enterprise
    # compliance / security officer would want for every query.
    if resp.trace is not None:
        _render_pipeline_trace(resp.trace)


def _render_pipeline_trace(t) -> None:
    """Render the 7-stage pipeline trace as a numbered checklist."""
    st.divider()
    st.subheader("Full pipeline trace (compliance audit)")
    st.caption(
        "Every stage the query passed through. This is what a security or "
        "compliance officer would see in the audit log for any single query."
    )

    # Stage 1 - identity
    with st.expander("**1. Identity resolution**", expanded=True):
        st.markdown(f"""
* **Email (from auth layer):** `{t.user_email}`
* **Resolved user_id:** `{t.user_id}`
* **Role:** `{t.user_role}`
* **Department:** `{t.user_department}`
* **Clearance level:** `{t.user_clearance}`
* **Accessible silos:** `{', '.join(t.accessible_departments) or '* (wildcard - exec)'}`
        """)
        st.caption(
            ":bulb: In production, `user_email` comes from a validated SSO/OAuth "
            "session token. The user never types it themselves."
        )

    # Stage 2 - routing
    with st.expander("**2. Query routing (intent classification)**", expanded=True):
        st.markdown(
            f"**Detected intents:** "
            + (", ".join(f"`{i}`" for i in t.detected_intents) or "_none_")
        )
        st.caption(
            "Router uses a keyword classifier first (fast, auditable) with "
            "an embedding-similarity fallback for queries the keywords miss. "
            "Detected intents are used to boost matching chunks during retrieval."
        )

    # Stage 3 - pre-filter
    with st.expander("**3. Pre-retrieval RBAC filter**", expanded=True):
        if t.chroma_where_clause is None:
            st.success(
                "No pre-filter applied - user has wildcard access (executive). "
                "ALL documents are eligible for retrieval."
            )
        else:
            st.markdown(
                "Documents NOT matching this filter are **never seen** by the "
                "vector search - blocked at the database layer:"
            )
            st.json(t.chroma_where_clause)
        st.caption(
            "This is the cheap, broad gate. It enforces clearance + department "
            "constraints at the vector-store level so denied content never enters "
            "the candidate pool. The fine-grained tag-policy checks run in stage 5."
        )

    # Stage 4 - retrieval
    with st.expander("**4. Vector search (retrieval)**", expanded=True):
        st.markdown(f"""
* **Embedding model:** `{t.embedding_model}`
* **Top-K requested:** `{t.top_k_requested}`
* **Candidates returned by ChromaDB:** `{t.candidates_retrieved}`
        """)
        st.caption(
            "Over-fetching beyond the final answer size so that some RBAC "
            "denials in stage 5 don't starve the generator of context."
        )

    # Stage 5 - post-filter
    with st.expander("**5. Post-retrieval RBAC check (defense-in-depth)**",
                     expanded=True):
        cols = st.columns(2)
        cols[0].metric("✅ Allowed", t.chunks_allowed)
        cols[1].metric("🔴 Denied", t.chunks_denied)
        if t.denied_reasons_summary:
            st.markdown("**Distinct denial reasons:**")
            for reason in t.denied_reasons_summary:
                st.markdown(f"* {reason}")
        else:
            st.success("No chunks were blocked at this stage.")
        st.caption(
            "Every surviving chunk is re-checked against the FULL policy set "
            "(including fine-grained tag rules like `salary_data`). If the "
            "stage-3 pre-filter ever drifts from the full policy, this layer "
            "still catches the leak. Both layers call the same predicate."
        )

    # Stage 6 - generation
    with st.expander("**6. Answer generation**", expanded=True):
        if t.refused_before_llm:
            st.error(
                f"🚫 LLM was **not called** - RBAC blocked all retrieved "
                f"context. Backend that *would have been* used: `{t.llm_backend}`."
            )
            st.caption(
                "Cheaper (no token spend) and safer (no chance the LLM "
                "'remembers' sensitive facts from its training data)."
            )
        else:
            st.markdown(f"""
* **Backend:** `{t.llm_backend}`
* **Model:** `{t.llm_model}`
* **Context chunks sent to LLM:** `{t.context_chunks_to_llm}` (out of `{t.chunks_allowed}` allowed)
* **Temperature:** `0.1` (low - we want grounded answers, not creativity)
            """)
            st.caption(
                "System prompt forbids using prior knowledge and requires "
                "bracketed citations for every fact."
            )

    # Stage 7 - response timing
    with st.expander("**7. Response & timing**", expanded=True):
        cols = st.columns(2)
        cols[0].metric("Total latency", f"{t.elapsed_ms:.0f} ms")
        cols[1].metric("Audit timestamp", t.timestamp_iso.replace("T", " "))
        st.caption(
            "End-to-end wall-clock from ask() entry to response return. "
            "In production this would be emitted to an append-only audit "
            "log alongside the per-chunk decisions from stage 5."
        )


# ---------------------------------------------------------------------------
# Trigger inference
# ---------------------------------------------------------------------------
# Two paths into inference:
#   - the user clicked Ask with a non-empty query, OR
#   - the user clicked a sample-query button (sets auto_submit).
should_run = (ask or auto_submit) and bool(query.strip())

# Debug panel - useful when something looks stuck. Hidden behind an expander
# so it doesn't clutter the normal demo flow.
with st.expander("🔧 Debug info (internal state)"):
    st.caption("These values reset on every rerun.")
    st.json({
        "selected_email": selected_email,
        "user_role": user.role,
        "user_clearance": user.clearance,
        "query_text": query,
        "ask_clicked": ask,
        "auto_submit": auto_submit,
        "should_run": should_run,
    })

if should_run:
    pipeline = get_pipeline()
    with st.spinner(f"Asking as {user.name}..."):
        resp = pipeline.ask(query.strip(), selected_email)
    render_response(resp)

elif ask and not query.strip():
    # User clicked Ask without typing anything - guide them.
    st.warning(
        "Please type a question in the box above, or click one of the "
        "sample queries on the left sidebar."
    )

elif not query.strip():
    # First load - show the help banner.
    st.info(
        "Pick a persona on the left, then type a question (or click a sample). "
        "Watch how the audit trail changes when you switch users -- denied "
        "chunks turn red, and refusals include the exact policy that blocked them."
    )
