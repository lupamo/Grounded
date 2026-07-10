# A RAG system with a visible retrival quality dashboard
A Q&A app where every answer shows its source and a live dashboard proves the retrival is actually accurate, not just "looks plausible".
### Features
- Hybrid retrieval: pgvector semantic search + Postgress full-text (BM25-style) search, merged
- Reranking step (cross-encoder or Cohere rerank) before generation
- Inline citations - every claim links to the exacr source chunk, highlighted.
- Live eval panel: faithfulness score (does the answer actially follow from retrived chunks?) and retrieval precision, computed per-query with LLM as  judge right in the UI.

## Stack
- Backend: Fast API, PostgreSQL + pgvector extension
- Retrieval: pgvector cosine search + Postgres tsvector full-text, reranked with Cohere's rerank APi (free tire) or local cross-encoder
- Generation: Claude API
- Frontend: Nextjs, streaming via SSE
- Eval: a lightweight LLN as judge score