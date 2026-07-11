-- Grounded — Step 2: Schema
--
-- Two tables:
--   documents  — one row per source file (e.g. "Data Protection Act.pdf")
--   chunks     — one row per retrievable chunk of that document
--
-- Each chunk carries TWO representations for retrieval:
--   embedding  (vector)    — for semantic/similarity search via pgvector
--   search_vec (tsvector)  — for exact keyword search via Postgres full-text
-- Hybrid retrieval later means: query both, merge, rerank.

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    source_path TEXT,                
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- bge-small-en-v1.5 outputs 384-dimensional embeddings.
-- If you swap embedding models later, this dimension MUST match the
-- model's output size or inserts will fail.
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,       
    content     TEXT NOT NULL,          
    embedding   vector(384),
    search_vec  tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for fast approximate nearest-neighbor search on embeddings.
-- IVFFlat is fine at this scale (hundreds to low thousands of chunks);
-- HNSW is the better choice once a project grows past that, but IVFFlat
-- is simpler to reason about while learning.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for fast full-text search.
CREATE INDEX IF NOT EXISTS chunks_search_vec_idx
    ON chunks USING gin (search_vec);

-- Speeds up "give me all chunks for this document" lookups.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);