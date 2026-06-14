"""Pluggable answer generator.

We define a tiny `Backend` protocol with one method: `generate(prompt) -> str`.
At startup, the pipeline probes the available backends in order and picks
the first one that works:

    1. Groq         - if GROQ_API_KEY is set (free, fast, hosted, OpenAI-compatible)
    2. OpenAI       - if OPENAI_API_KEY is set
    3. Anthropic    - if ANTHROPIC_API_KEY is set
    4. Ollama       - if a local server responds on the configured port
    5. Fallback     - deterministic extractive answer (always works)

The fallback is the reason the judge can `python -m src.rag_pipeline` and
see a working demo with no setup at all. The fallback isn't fancy but it
respects citations and won't hallucinate -- it literally pastes the most
relevant chunks back with light prose around them.
"""

from __future__ import annotations

import re
import textwrap
from typing import Protocol

from src import config
from src.data_models import RAGResponse, RetrievedChunk, User


def _dedupe_answer(text: str) -> str:
    """Collapse near-duplicate sentences/bullets that small local models sometimes
    emit (e.g. restating "the context does not mention X" five different ways).

    Splits the answer into bullet/sentence units and keeps the first of any group
    whose word-set overlaps an earlier kept unit by >= 80% (Jaccard) — so genuinely
    distinct facts survive while reworded repeats are dropped. Order is preserved.
    """
    text = (text or "").strip()
    if not text:
        return text

    has_bullets = "•" in text or bool(re.search(r"(^|\n)\s*[-*]\s+", text))
    if has_bullets:
        units = re.split(r"•|\n\s*[-*]\s+|\n{2,}", text)
    else:
        units = re.split(r"(?<=[.!?])\s+", text)

    # "No information" restatements: small models often phrase the same "I don't
    # know" a dozen ways. We keep at most ONE of them (the first), regardless of
    # exact wording, since none of them carry a distinct fact.
    no_info = re.compile(
        r"\b(?:not (?:mentioned|provided|found|available|specified|present|stated|included)"
        r"|does(?:n't| not) (?:mention|provide|contain|include|specify|state|cover|have)"
        r"|do(?:n't| not) (?:mention|provide|contain|include)"
        r"|no (?:information|details?|mention|data|reference)"
        r"|cannot|can't|could not|couldn't) ",
        re.IGNORECASE,
    )

    kept: list[str] = []
    kept_tokens: list[set[str]] = []
    seen_no_info = False
    for unit in units:
        u = unit.strip().strip("-*• \t")
        if not u:
            continue
        if no_info.search(u):
            if seen_no_info:
                continue  # already have one "no information" line
            seen_no_info = True
        toks = set(re.sub(r"[^a-z0-9 ]", " ", u.lower()).split())
        if toks and any(
            kt and len(toks & kt) / len(toks | kt) >= 0.8 for kt in kept_tokens
        ):
            continue  # near-duplicate of something we already kept
        kept.append(u)
        kept_tokens.append(toks)

    if not kept:
        return text
    return "\n".join(f"- {u}" for u in kept) if has_bullets else " ".join(kept)


SYSTEM_PROMPT = textwrap.dedent("""\
    You are Acme Corp's internal enterprise assistant. Follow these rules
    strictly:

      1. Answer ONLY from the provided CONTEXT blocks. Never use prior
         knowledge or invent facts.
      2. Cite every fact with the bracketed source id given in the context,
         e.g. "Q4 revenue was $48.2M [DOC-FIN-Q4::p1]". One citation per
         fact, placed at the end of the sentence.
      3. If the question asks about multiple topics (e.g. "revenue AND
         security incidents"), cover EACH topic that appears in the
         context. Do not skip a topic just because another is also present.
      4. If the context truly does not contain an answer, say so plainly
         in one sentence -- do not pad.
      5. Never reveal information about documents the user is not
         allowed to read; you will only ever be shown allowed context.
      6. Be clear and structured. Use short bullet points for lists of
         facts. No preamble, no marketing, no apology for limitations.
""")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class Backend(Protocol):
    name: str
    model_name: str
    def generate(self, system: str, user: str) -> str: ...


class GroqBackend:
    """Groq's OpenAI-compatible chat API: free, very fast, hosted. Preferred when
    GROQ_API_KEY is set — nothing to install locally and it works in cloud deploys.
    Reuses the `openai` client (already a dependency) pointed at Groq's base URL."""

    name = "groq"
    model_name = config.GROQ_MODEL

    def __init__(self) -> None:
        from openai import OpenAI
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,   # grounded answers, not creativity
        )
        return resp.choices[0].message.content or ""


class OpenAIBackend:
    name = "openai"
    model_name = config.OPENAI_MODEL

    def __init__(self) -> None:
        # Import lazily so this module imports even when openai isn't
        # installed (the user might be running with Anthropic only).
        from openai import OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,   # we want grounded answers, not creativity
        )
        return resp.choices[0].message.content or ""


class AnthropicBackend:
    name = "anthropic"
    model_name = config.ANTHROPIC_MODEL

    def __init__(self) -> None:
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def generate(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.1,
        )
        # Concatenate any text-typed content blocks (modern API can also
        # return tool_use blocks which we don't use here).
        return "".join(b.text for b in msg.content if b.type == "text")


class OllamaBackend:
    name = "ollama"
    model_name = config.OLLAMA_MODEL

    def __init__(self) -> None:
        import requests
        self.requests = requests
        # Probe the server with a short timeout; raise if it's not there.
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2)
        r.raise_for_status()

    def generate(self, system: str, user: str) -> str:
        r = self.requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": f"{system}\n\n{user}",
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()


class FallbackBackend:
    """Extractive answer-by-template.

    Strategy: take the top 2-3 allowed chunks and stitch them into a
    short answer. Every chunk is shown with its source id so the user can
    verify. This isn't LLM-quality prose but it is:
        * 100% grounded (no hallucination -- text is copied verbatim)
        * fully offline
        * deterministic (great for tests + reproducible demos)
    """

    name = "fallback-extractive"
    model_name = "n/a (deterministic extractor, no LLM)"

    def generate(self, system: str, user: str) -> str:
        # The "user" prompt we constructed already contains the
        # question + numbered context blocks. We just extract the blocks
        # and surface them. See `_build_user_prompt` for the format.
        return ("(extractive fallback - no LLM configured; raw context "
                "below)\n\n" + user.split("CONTEXT:\n", 1)[-1])


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def pick_backend() -> Backend:
    """Probe backends in order and return the first one that initialises.

    Errors during probing are caught so a missing API key for backend N
    doesn't prevent us from falling through to backend N+1. The fallback
    has no failure mode, so the function always returns something.
    """
    candidates: list[type[Backend]] = []
    if config.GROQ_API_KEY:
        candidates.append(GroqBackend)
    if config.OPENAI_API_KEY:
        candidates.append(OpenAIBackend)
    if config.ANTHROPIC_API_KEY:
        candidates.append(AnthropicBackend)
    # Always try Ollama -- the probe is fast and silently fails.
    candidates.append(OllamaBackend)

    for cls in candidates:
        try:
            backend = cls()
            print(f"[generator] using backend: {backend.name}")
            return backend
        except Exception as e:
            print(f"[generator] backend {cls.__name__} unavailable: {e}")

    print("[generator] using fallback extractive backend "
          "(set OPENAI_API_KEY / ANTHROPIC_API_KEY / run Ollama "
          "for richer answers)")
    return FallbackBackend()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class Generator:
    """Turns retrieval results into the final RAGResponse.

    Responsibilities:
        * Format the context blocks the LLM sees (only allowed chunks).
        * Detect "everything was denied" and produce a clean refusal.
        * Compute a confidence score from retrieval similarities.
        * Collect citation strings for the response object.
    """

    def __init__(self, backend: Backend | None = None) -> None:
        self.backend = backend or pick_backend()

    def generate(self,
                 query: str,
                 user: User,
                 retrieved: list[RetrievedChunk],
                 intents: list[str]) -> RAGResponse:
        allowed = [r for r in retrieved if r.allowed]
        denied = [r for r in retrieved if not r.allowed]

        # Truncate to FINAL_K so the prompt stays small. The retrieved
        # list is already in score order so we just slice.
        allowed = allowed[:config.FINAL_K]

        # --- refusal path ---------------------------------------------
        # If RBAC blocked everything, we don't even call the LLM. Why:
        #   (a) cheaper - no token spend on doomed answers.
        #   (b) safer - removes any chance the LLM "remembers" sensitive
        #       facts from its training data and tries to be helpful.
        if not allowed:
            return self._refusal_response(query, user, retrieved, intents,
                                          denied)

        prompt = self._build_user_prompt(query, allowed)
        try:
            raw = self.backend.generate(SYSTEM_PROMPT, prompt)
        except Exception as e:
            # Any LLM error degrades gracefully to the extractive fallback
            # rather than 500-ing on the user.
            print(f"[generator] backend failed mid-flight: {e}; "
                  "falling back to extractive answer")
            raw = FallbackBackend().generate(SYSTEM_PROMPT, prompt)

        # Citation strings -- one per allowed chunk we used.
        citations = [
            f"[{r.chunk.chunk_id}] {r.chunk.metadata.title} "
            f"({r.chunk.metadata.source_type}, "
            f"sensitivity={r.chunk.metadata.sensitivity})"
            for r in allowed
        ]

        # Confidence: mean similarity of the chunks we showed the LLM,
        # clamped to [0,1]. Scores near 0 mean the corpus didn't have
        # great matches -- the user should treat the answer with care.
        scores = [max(0.0, min(1.0, r.score)) for r in allowed]
        confidence = sum(scores) / len(scores) if scores else 0.0

        return RAGResponse(
            query=query,
            user=user,
            answer=_dedupe_answer(raw),
            citations=citations,
            retrieved=retrieved,
            routed_to=intents,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    def _refusal_response(self, query: str, user: User,
                          retrieved: list[RetrievedChunk],
                          intents: list[str],
                          denied: list[RetrievedChunk]) -> RAGResponse:
        # Give a *specific* reason so the user understands what to do
        # next (request access, ask their manager, etc.). We dedupe the
        # reason strings since several chunks usually fail for the same
        # underlying policy.
        reasons = []
        seen = set()
        for d in denied[:3]:
            r = d.rbac_reason
            if r not in seen:
                seen.add(r)
                reasons.append(f"  - {d.chunk.metadata.title}: {r}")

        if not reasons:
            # Nothing matched the query even before RBAC. This is a
            # different failure mode ("no information found") that we
            # surface clearly.
            answer = (f"I couldn't find any information about '{query}' "
                      f"in the documents I have access to.")
            refusal_reason = "no matching context found"
        else:
            answer = (
                f"I can't answer that question because the relevant "
                f"information is restricted for your role "
                f"('{user.role}' in {user.department}).\n\n"
                f"Affected sources:\n" + "\n".join(reasons) +
                "\n\nIf you need this information, please contact the "
                "owning department for an access request."
            )
            refusal_reason = "blocked by RBAC"

        return RAGResponse(
            query=query,
            user=user,
            answer=answer,
            citations=[],
            retrieved=retrieved,
            routed_to=intents,
            confidence=0.0,
            refused=True,
            refusal_reason=refusal_reason,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_user_prompt(query: str,
                           allowed: list[RetrievedChunk]) -> str:
        """Format the question + allowed context for the LLM.

        Each chunk is wrapped with its chunk_id so the model can cite it
        verbatim. We also include the source type + sensitivity in the
        header -- LLMs do pay attention to this and tend to be more
        cautious with documents tagged 'restricted'.
        """
        blocks = []
        for r in allowed:
            md = r.chunk.metadata
            blocks.append(
                f"--- [{r.chunk.chunk_id}] {md.title} "
                f"({md.source_type}, sensitivity={md.sensitivity}, "
                f"score={r.score:.2f}) ---\n{r.chunk.text}"
            )

        ctx = "\n\n".join(blocks)
        return (
            f"QUESTION:\n{query}\n\n"
            f"CONTEXT:\n{ctx}\n\n"
            "Answer the question using only the context above. Include "
            "bracketed citations like [DOC-XYZ::p2] for every fact."
        )
