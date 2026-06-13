"""Load + chunk multi-format enterprise data.

Each loader is responsible for ONE source type. They all output a uniform
list[Chunk] so the rest of the pipeline is format-agnostic.

Why chunk at all?
    Embeddings work best when each vector represents a single coherent idea.
    A 50-page PDF embedded as one vector is useless for search; 50 paragraph
    -sized chunks let us pinpoint the exact passage that answered a query.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pypdf import PdfReader

from src import config
from src.data_models import Chunk, DocumentMetadata


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
def _chunk_text(text: str,
                chunk_size: int = config.CHUNK_SIZE,
                overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows respecting paragraph boundaries.

    Strategy: split on blank lines first (paragraphs are semantically
    coherent), then greedily pack paragraphs into ~chunk_size buckets.
    When a single paragraph exceeds chunk_size we fall back to a sliding
    window with `overlap` characters of carry-over.

    The overlap matters because retrieval cuts off at boundaries: without
    it, a sentence that straddles two chunks gets weak similarity scores
    on both sides.
    """
    if not text.strip():
        return []

    # Normalise whitespace and split on blank lines.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            # Flush whatever we had buffered, then sliding-window the giant
            # paragraph so we don't drop content.
            if buf:
                chunks.append(buf.strip())
                buf = ""
            for i in range(0, len(para), chunk_size - overlap):
                chunks.append(para[i:i + chunk_size])
            continue

        # Common case: keep packing paragraphs until we'd exceed chunk_size.
        if len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}" if buf else para
        else:
            chunks.append(buf.strip())
            buf = para

    if buf.strip():
        chunks.append(buf.strip())
    return chunks


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------
def _load_pdf(path: Path, meta: DocumentMetadata) -> list[Chunk]:
    """Extract text page-by-page then chunk.

    pypdf can be patchy on heavily-formatted PDFs, but ours are generated
    from ReportLab so this is reliable. For real-world ingestion you would
    want OCR fallback (Tesseract) for scanned docs.
    """
    reader = PdfReader(str(path))
    full_text = "\n\n".join((page.extract_text() or "") for page in reader.pages)

    out: list[Chunk] = []
    for i, text in enumerate(_chunk_text(full_text)):
        out.append(Chunk(
            chunk_id=f"{meta.doc_id}::p{i}",
            text=text,
            metadata=meta,
            chunk_index=i,
        ))
    return out


def _load_csv(path: Path, meta: DocumentMetadata) -> list[Chunk]:
    """Each row becomes its own chunk.

    Doing one chunk per row keeps the salary-lookup case crisp: when
    Alice asks "what is Bob's salary?" we want to retrieve a single row
    (Bob's), not a 200-row dump.

    The text representation flattens the row to "col1: val1 | col2: val2 ..."
    which embeds better than raw CSV (more like natural language).
    """
    out: list[Chunk] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            line = " | ".join(f"{k}: {v}" for k, v in row.items())
            out.append(Chunk(
                chunk_id=f"{meta.doc_id}::row{i}",
                text=line,
                metadata=meta,
                chunk_index=i,
            ))
    return out


def _load_json_log(path: Path, meta: DocumentMetadata) -> list[Chunk]:
    """One chunk per JSON event.

    We flatten each event to a readable sentence-ish form because the
    embedding model is trained on natural English, not nested JSON. For
    deeply nested logs you'd want a recursive flattener; here the schema
    is shallow enough that we can do it inline.
    """
    events = json.loads(path.read_text())
    if not isinstance(events, list):
        events = [events]

    out: list[Chunk] = []
    for i, ev in enumerate(events):
        parts = []
        for k, v in ev.items():
            if isinstance(v, list):
                v = ", ".join(map(str, v)) if v else "(none)"
            parts.append(f"{k}: {v}")
        text = " | ".join(parts)
        out.append(Chunk(
            chunk_id=f"{meta.doc_id}::ev{i}",
            text=text,
            metadata=meta,
            chunk_index=i,
        ))
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def load_all() -> list[Chunk]:
    """Walk the data directories and return every chunk.

    The mapping of filename -> metadata comes from
    `data/policies/documents_metadata.json` (built by generate_data.py).
    Keeping it external means a security/compliance person can re-classify
    a document without touching code.
    """
    meta_path = config.POLICIES_DIR / "documents_metadata.json"
    meta_raw = json.loads(meta_path.read_text())

    # Walk every data subdir; pick the loader based on suffix.
    chunks: list[Chunk] = []
    for folder in (config.DOCS_DIR, config.DB_DIR, config.LOGS_DIR):
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.name not in meta_raw:
                # Skip files we haven't classified -- safer to ignore than
                # to ingest with a default sensitivity that could be wrong.
                print(f"[ingest] skipping unclassified file: {path.name}")
                continue

            raw = meta_raw[path.name]
            md = DocumentMetadata(
                doc_id=raw["doc_id"],
                source_path=str(path),
                source_type=path.suffix.lstrip(".").lower(),
                department=raw["department"],
                sensitivity=raw["sensitivity"],
                title=raw["title"],
                tags=raw.get("tags", []),
            )

            if path.suffix.lower() == ".pdf":
                chunks.extend(_load_pdf(path, md))
            elif path.suffix.lower() == ".csv":
                chunks.extend(_load_csv(path, md))
            elif path.suffix.lower() == ".json":
                chunks.extend(_load_json_log(path, md))
            else:
                print(f"[ingest] no loader for: {path.name}")

    return chunks
