"""Streamlit UI for the Enterprise RAG demo.

Run with:
    streamlit run streamlit_app.py

The UI is a thin presentation layer on top of `src.rag_pipeline.RAGPipeline`.
It exists to make the RBAC story visible -- the same query asked by different
personas produces visibly different responses, with the allowed/denied audit
trail rendered, and an optional side-by-side compare so the contrast between
two roles lands in a single screen.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Make `src` importable when streamlit is invoked from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.data_models import RAGResponse
from src.rag_pipeline import RAGPipeline
from src.rbac import RBACEngine


# ---------------------------------------------------------------------------
# Page setup + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise RAG - RBAC Demo",
    layout="wide",
    page_icon=":lock:",
)

# A little CSS so the demo feels intentional: an accent colour, a result banner,
# and citation "chips" that read as references rather than noise in the prose.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.4rem; }
      .stButton > button[kind="primary"] { background:#4f46e5; border:0; font-weight:600; }
      .stButton > button[kind="primary"]:hover { background:#4338ca; }

      .banner { padding:.7rem 1rem; border-radius:.6rem; margin:.1rem 0 1rem;
                font-size:1.0rem; border:1px solid transparent; line-height:1.45; }
      .banner.ok      { background:rgba(34,197,94,.12);  border-color:rgba(34,197,94,.45); }
      .banner.refused { background:rgba(239,68,68,.12);  border-color:rgba(239,68,68,.5); }

      /* inline citation chip */
      .cite { font-size:.7em; background:rgba(99,102,241,.18); color:#a5b4fc;
              border:1px solid rgba(99,102,241,.4); padding:0 .35em; border-radius:.45em;
              margin:0 .12em; white-space:nowrap; vertical-align:super; font-weight:600; }
      .citation { font-size:.9rem; padding:.12rem 0; }
      .persona-head { font-weight:700; font-size:1.05rem; padding:.25rem 0;
                      border-bottom:2px solid rgba(99,102,241,.45); margin-bottom:.7rem; }
    </style>
    """,
    unsafe_allow_html=True,
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
# Rendering helpers
# ---------------------------------------------------------------------------
# Chunk ids look like DOC-HR-HANDBOOK::p0, DB-EMPLOYEES::row3, LOG-SEC-AUDIT::ev1.
_CITE_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9\-]*::[A-Za-z0-9]+)\]")


def _chips(text: str) -> str:
    """Render bracketed citation ids as subtle inline chips instead of raw noise."""
    return _CITE_RE.sub(r"<span class='cite'>\1</span>", text or "")


def _banner(resp: RAGResponse) -> None:
    """A single colour-coded headline that states the access decision up front."""
    if resp.refused:
        st.markdown(
            f"<div class='banner refused'>🚫 <b>Access refused</b> for "
            f"<b>{resp.user.name}</b> · <code>{resp.user.role}</code> in "
            f"{resp.user.department} — {resp.refusal_reason}</div>",
            unsafe_allow_html=True,
        )
    else:
        routed = ", ".join(resp.routed_to) or "—"
        st.markdown(
            f"<div class='banner ok'>✅ <b>Answered</b> for <b>{resp.user.name}</b> · "
            f"<code>{resp.user.role}</code> · routed to <b>{routed}</b> · "
            f"confidence {resp.confidence:.0%}</div>",
            unsafe_allow_html=True,
        )


def _confidence_bar(resp: RAGResponse) -> None:
    if resp.refused:
        st.progress(1.0, text="RBAC decision: deterministic (rule-based, no statistical uncertainty)")
    else:
        bar = min(max(resp.confidence, 0.0), 1.0)
        st.progress(bar, text=f"Retrieval confidence: {bar:.0%}")


def _answer_and_citations(resp: RAGResponse) -> None:
    st.subheader("Answer")
    if resp.answer:
        st.markdown(_chips(resp.answer), unsafe_allow_html=True)
    if resp.citations:
        st.subheader("Citations")
        for c in resp.citations:
            st.markdown(f"<div class='citation'>{_chips(c)}</div>", unsafe_allow_html=True)


def _audit_trail(resp: RAGResponse, *, expanded: bool) -> None:
    """The explainability money shot: per-chunk ALLOW/DENY with reasons."""
    n_allowed = sum(r.allowed for r in resp.retrieved)
    n_total = len(resp.retrieved)
    # Label includes the persona name so two columns in compare mode never collide
    # on an identical expander label (Streamlit keys expanders by their label).
    with st.expander(
        f"📋 Retrieval audit trail · {resp.user.name} — {n_allowed} of {n_total} chunks "
        f"visible ({n_total - n_allowed} blocked by RBAC)",
        expanded=expanded,
    ):
        st.caption(
            "Every candidate chunk the retriever surfaced, with its similarity score and the "
            "access-control decision applied. Denied chunks were never shown to the language model."
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
                f"{badge}  &nbsp;  score=`{r.score:+.2f}`  &nbsp;  `{r.chunk.chunk_id}`  &nbsp;  "
                f"({r.chunk.metadata.source_type}, sensitivity={sens_colour})"
            )
            st.caption(f"&nbsp;&nbsp;&nbsp;reason: {r.rbac_reason}")
            preview = r.chunk.text[:240] + ("..." if len(r.chunk.text) > 240 else "")
            st.code(preview, language=None)


def _render_pipeline_trace(t) -> None:
    """The 7-stage end-to-end audit record. Shown only when the user opts in."""
    st.caption(
        "Every stage the query passed through — the record a security / compliance officer "
        "would see in the audit log for any single query."
    )

    with st.expander("**1. Identity resolution**", expanded=True):
        st.markdown(f"""
* **Email (from auth layer):** `{t.user_email}`
* **Resolved user_id:** `{t.user_id}`
* **Role:** `{t.user_role}`
* **Department:** `{t.user_department}`
* **Clearance level:** `{t.user_clearance}`
* **Accessible silos:** `{', '.join(t.accessible_departments) or '* (wildcard - exec)'}`
        """)
        st.caption(":bulb: In production, `user_email` comes from a validated SSO/OAuth "
                   "session token. The user never types it themselves.")

    with st.expander("**2. Query routing (intent classification)**", expanded=True):
        st.markdown("**Detected intents:** "
                    + (", ".join(f"`{i}`" for i in t.detected_intents) or "_none_"))
        st.caption("Router uses a keyword classifier first (fast, auditable) with an "
                   "embedding-similarity fallback for queries the keywords miss.")

    with st.expander("**3. Pre-retrieval RBAC filter**", expanded=True):
        if t.chroma_where_clause is None:
            st.success("No pre-filter applied - user has wildcard access (executive). "
                       "ALL documents are eligible for retrieval.")
        else:
            st.markdown("Documents NOT matching this filter are **never seen** by the "
                        "vector search - blocked at the database layer:")
            st.json(t.chroma_where_clause)
        st.caption("The cheap, broad gate: clearance + department constraints enforced at the "
                   "vector-store level so denied content never enters the candidate pool.")

    with st.expander("**4. Vector search (retrieval)**", expanded=True):
        st.markdown(f"""
* **Embedding model:** `{t.embedding_model}`
* **Top-K requested:** `{t.top_k_requested}`
* **Candidates returned by ChromaDB:** `{t.candidates_retrieved}`
        """)
        st.caption("Over-fetching beyond the final answer size so RBAC denials in stage 5 "
                   "don't starve the generator of context.")

    with st.expander("**5. Post-retrieval RBAC check (defense-in-depth)**", expanded=True):
        cols = st.columns(2)
        cols[0].metric("✅ Allowed", t.chunks_allowed)
        cols[1].metric("🔴 Denied", t.chunks_denied)
        if t.denied_reasons_summary:
            st.markdown("**Distinct denial reasons:**")
            for reason in t.denied_reasons_summary:
                st.markdown(f"* {reason}")
        else:
            st.success("No chunks were blocked at this stage.")
        st.caption("Every surviving chunk is re-checked against the FULL policy set (including "
                   "fine-grained tag rules like `salary_data`). Both layers call the same predicate.")

    with st.expander("**6. Answer generation**", expanded=True):
        if t.refused_before_llm:
            st.error(f"🚫 LLM was **not called** - RBAC blocked all retrieved context. "
                     f"Backend that *would have been* used: `{t.llm_backend}`.")
            st.caption("Cheaper (no token spend) and safer (no chance the LLM 'remembers' "
                       "sensitive facts from its training data).")
        else:
            st.markdown(f"""
* **Backend:** `{t.llm_backend}`
* **Model:** `{t.llm_model}`
* **Context chunks sent to LLM:** `{t.context_chunks_to_llm}` (out of `{t.chunks_allowed}` allowed)
* **Temperature:** `0.1` (low - we want grounded answers, not creativity)
            """)
            st.caption("System prompt forbids using prior knowledge and requires bracketed "
                       "citations for every fact.")

    with st.expander("**7. Response & timing**", expanded=True):
        cols = st.columns(2)
        cols[0].metric("Total latency", f"{t.elapsed_ms:.0f} ms")
        cols[1].metric("Audit timestamp", t.timestamp_iso.replace("T", " "))
        st.caption("End-to-end wall-clock from ask() entry to response return. In production "
                   "this is emitted to an append-only audit log alongside stage-5 decisions.")


def render_response(resp: RAGResponse) -> None:
    """Full single-column render: banner → answer → citations → audit → (opt) trace."""
    _banner(resp)
    _confidence_bar(resp)
    st.divider()
    _answer_and_citations(resp)
    _audit_trail(resp, expanded=True)

    # The full pipeline trace is secondary detail — gated behind a toggle so it
    # doesn't bury the answer under a long scroll. Opt in to see all 7 stages.
    if resp.trace is not None:
        st.divider()
        if st.toggle("🔬 Show full pipeline trace (compliance audit)", key="trace_single"):
            _render_pipeline_trace(resp.trace)


def render_compare_column(resp: RAGResponse) -> None:
    """Compact render for the side-by-side compare view (one persona per column)."""
    st.markdown(
        f"<div class='persona-head'>{resp.user.name} · {resp.user.role}</div>",
        unsafe_allow_html=True,
    )
    _banner(resp)
    _answer_and_citations(resp)
    n_allowed = sum(r.allowed for r in resp.retrieved)
    n_total = len(resp.retrieved)
    st.caption(f"🔓 {n_allowed}/{n_total} chunks visible to this role"
               + ("" if resp.refused else f" · confidence {resp.confidence:.0%}"))
    _audit_trail(resp, expanded=False)


# ---------------------------------------------------------------------------
# Sidebar - persona switcher
# ---------------------------------------------------------------------------
rbac = get_rbac()

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
    # Clicking a sample button fills the query box AND auto-submits. We do NOT call
    # st.rerun() here: the button click already triggers a rerun, and the main panel
    # (rendered after the sidebar) picks up the new query + auto_submit on that same
    # pass. Calling st.rerun() before the main-panel widgets are instantiated would
    # make Streamlit garbage-collect their state — silently switching off compare mode.
    for q in sample_queries:
        if st.button(q, use_container_width=True, key=f"sample_{hash(q)}"):
            st.session_state["query_text"] = q
            st.session_state["auto_submit"] = True


# ---------------------------------------------------------------------------
# Main panel - query + response
# ---------------------------------------------------------------------------
st.title("Ask the enterprise assistant")
st.caption(
    "The same question may yield different answers depending on who is asking. Each retrieved "
    "chunk is independently access-checked; chunks you can't see are blocked and the reason is logged."
)

if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""

query = st.text_input(
    "Your question",
    key="query_text",
    placeholder="e.g. What is the parental leave policy?",
    label_visibility="collapsed",
)

# Action row: Ask + the compare toggle. When compare is on, choose a second role.
ctrl = st.columns([1, 2, 3])
ask = ctrl[0].button("Ask", type="primary", use_container_width=True)
# Explicit keys so the toggle + second-role selection persist across the rerun that
# a sample-query button or the Ask button triggers (otherwise compare mode would
# silently switch off the moment you run a query).
compare = ctrl[1].toggle("🔀 Compare two roles", key="compare_mode",
                         help="Ask the same question as two roles and see the answers side by side.")
compare_email = None
if compare:
    others = [e for e in sidebar_options if e != selected_email]
    compare_email = ctrl[2].selectbox(
        "Second role",
        options=others,
        format_func=lambda e: sidebar_options[e],
        index=0,
        label_visibility="collapsed",
        key="compare_email",
    )

# Auto-submit flag set by sidebar sample-query buttons.
auto_submit = st.session_state.pop("auto_submit", False)

# Two paths into inference: explicit Ask, or a sample-query button (auto_submit).
should_run = (ask or auto_submit) and bool(query.strip())

# Debug panel - hidden behind an expander so it doesn't clutter the demo flow.
with st.expander("🔧 Debug info (internal state)"):
    st.caption("These values reset on every rerun.")
    st.json({
        "selected_email": selected_email,
        "compare": compare,
        "compare_email": compare_email,
        "user_role": user.role,
        "user_clearance": user.clearance,
        "query_text": query,
        "ask_clicked": ask,
        "auto_submit": auto_submit,
        "should_run": should_run,
    })

if should_run:
    pipeline = get_pipeline()
    if compare and compare_email:
        with st.spinner("Asking as both roles..."):
            resp_a = pipeline.ask(query.strip(), selected_email)
            resp_b = pipeline.ask(query.strip(), compare_email)
        st.caption("Same question, two roles — note how the access decision and visible "
                   "chunks differ.")
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            render_compare_column(resp_a)
        with col_b:
            render_compare_column(resp_b)
    else:
        with st.spinner(f"Asking as {user.name}..."):
            resp = pipeline.ask(query.strip(), selected_email)
        render_response(resp)

elif ask and not query.strip():
    st.warning("Please type a question in the box above, or click one of the sample queries "
               "on the left sidebar.")

elif not query.strip():
    st.info(
        "Pick a persona on the left, then type a question (or click a sample). Watch how the "
        "audit trail changes when you switch users — denied chunks turn red, and refusals "
        "include the exact policy that blocked them. Try **🔀 Compare two roles** to see the "
        "same question answered for two roles side by side."
    )
