"""
Retrieval pipeline
combines two independent retrieval paths over the same chunk table:
1. semantic search - pgvector cosine similarity on embeddings
2. Full-text search - postgres tsvector/tsquery keyword matching
Usage:
    python retrieve.py "What's the difference between extensions and functions?"
"""

import argparse
import os
import sys

from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SEMANTIC_CANDIDATES = 15
FULLTEXT_CANDIDATES = 15
FINAL_TOP_K = 5

def semantic_search(cur, query_embedding, limit: int) -> list[dict]:
	"""
	Nearest neighbor search via pgvector cosine distance (<=> operator).
	Lower distance = more similar, so we order ascending
	"""
	cur.execute(
		"""
		SELECT id, document_id, chunk_index, content,
			1 - (embedding <=> %s) AS similarity
		FROM chunks
		ORDER BY embedding <=> %s
		LIMIT %s;
		""",
		(query_embedding, query_embedding, limit),
	)
	cols = ["id", "document_id", "chunk_index", "content", "similarity"]
	return [dict(zip(cols, row)) for row in cur.fetchall()]

def fulltext_search(cur, query: str, limit:int) -> list[dict]:
	"""
	Keyword search via Postgres full-text search, ranked by ts_rank.
    plainto_tsquery handles natural-language input without requiring the
    caller to know tsquery syntax (AND/OR/& etc.).
	"""
	cur.execute(
		"""
		SELECT id, document_id, chunk_index, content,
			ts_rank_cd(search_vec, plainto_tsquery('english', %s)) AS rank
		FROM chunks
		WHERE search_vec @@ plainto_tsquery('english', %s)
		ORDER BY rank DESC
		LIMIT %s;
		""",
		(query, query, limit),
	)
	cols = ["id", "document_id", "chunk_index", "content", "rank"]
	return [dict(zip(cols, row)) for row in cur.fetchall()]

def merge_candidates(semantic_results: list[dict], fulltext_results: list[dict]) -> list[dict]:
	"""
	Merge the two candidate lists, deduplicating by chunk id.
	We keep the semantic search score and add the full-text rank if present.
	"""
	merged = {}
	for r in semantic_results:
		merged[r["id"]] = {**r, "sources": ["semantic"]}
	for r in fulltext_results:
		if r["id"] in merged:
			merged[r["id"]]["sources"].append("fulltext")
		else:
			merged[r["id"]] = {**r, "sources": ["fulltext"]}
	return list(merged.values())

def rerank(query: str, candidates: list[dict], reranker: CrossEncoder, top_k: int) -> list[dict]:
    """Cross-encoders score a (query, passage) pair jointly, which is far
    more accurate than comparing separately-computed embeddings — the
    tradeoff is it's too slow to run over an entire corpus, hence using it
    only on the smaller candidate set retrieval already narrowed down."""
    pairs = [(query, c["content"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
 
 
def retrieve(query: str, embed_model: SentenceTransformer, reranker: CrossEncoder, conn) -> list[dict]:
    query_embedding = embed_model.encode(query, normalize_embeddings=True)
 
    with conn.cursor() as cur:
        semantic_results = semantic_search(cur, query_embedding, SEMANTIC_CANDIDATES)
        fulltext_results = fulltext_search(cur, query, FULLTEXT_CANDIDATES)
 
    candidates = merge_candidates(semantic_results, fulltext_results)
    return rerank(query, candidates, reranker, FINAL_TOP_K)
 
 
def main():
    parser = argparse.ArgumentParser(description="Hybrid retrieval + reranking over the Grounded knowledge base.")
    parser.add_argument("query", help="The question to search for.")
    args = parser.parse_args()
 
    if not DATABASE_URL:
        print("DATABASE_URL not set. Copy .env.example to .env first.")
        sys.exit(1)
 
    print("Loading models (embedding + reranker)...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
 
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
 
    print(f"\nQuery: {args.query}\n")
    results = retrieve(args.query, embed_model, reranker, conn)
 
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (rerank_score={r['rerank_score']:.3f}, found via {'+'.join(r['sources'])}) ---")
        print(r["content"][:300].replace("\n", " "))
        print()
 
    conn.close()
 
 
if __name__ == "__main__":
    main()
 
