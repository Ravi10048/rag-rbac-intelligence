"""Enterprise RAG package.

Top-level modules:
    config        - paths, tuning knobs, environment lookups
    data_models   - typed containers used across the pipeline
    rbac          - access-control engine (the security boundary)
    ingestion     - load + chunk PDFs / CSVs / JSON logs
    embeddings    - thin wrapper around sentence-transformers
    vector_store  - ChromaDB persistence + metadata filtering
    router        - lightweight query intent classifier
    retriever     - hybrid retrieval with RBAC pre/post filtering
    generator     - pluggable LLM backend (OpenAI / Anthropic / Ollama / fallback)
    rag_pipeline  - orchestrates everything end-to-end
"""
