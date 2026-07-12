"""
generation with forced citations.

Retrieves relevant chunks (step 4's hybrid search + rerank), then asks the
model to answer using ONLY those chunks, citing which chunk[s] support each
claim with bracketed numbers like [1], [2]. This is what makes an answer
checkable rather than just plausible-sounding.

Usage:
    python generate.py "What's the difference between extensions and functions?"
"""
import argparse
import os
import sys

from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

from retrieve import retrieve, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a research assistant. Answer the user's question using ONLY the numbered source excerpts provided below. Follow these rules strictly:

1. Every factual claim in your answer must be followed by a citation like [1] or [2][3] pointing to the source excerpt(s) that support it.
2. If the excerpts don't contain enough information to answer the question, say so explicitly — do not guess or use outside knowledge.
3. Keep the answer concise and directly responsive to the question.
4. Do not fabricate citation numbers that don't correspond to a provided excerpt."""


def build_context_block(chunks: list[dict]) -> str:
    """Formats retrieved chunks as numbered sources for the prompt, matching
    the citation numbers we'll ask the model to use."""
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] {c['content']}")
    return "\n\n".join(lines)


def generate_answer(query: str, chunks: list[dict], groq_client: Groq) -> str:
    context_block = build_context_block(chunks)
    user_message = f"Source excerpts:\n\n{context_block}\n\nQuestion: {query}"

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Ask a grounded, cited question over the Grounded knowledge base.")
    parser.add_argument("query", help="The question to ask.")
    args = parser.parse_args()

    if not DATABASE_URL:
        print("DATABASE_URL not set. Copy .env.example to .env first.")
        sys.exit(1)
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Get a free key at console.groq.com and add it to .env.")
        sys.exit(1)

    print("Loading models...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    groq_client = Groq()

    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)

    print(f"\nQuery: {args.query}\n")
    print("Retrieving relevant chunks...")
    chunks = retrieve(args.query, embed_model, reranker, conn)

    print("Generating answer...\n")
    answer = generate_answer(args.query, chunks, groq_client)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)
    for i, c in enumerate(chunks, 1):
        preview = c["content"][:150].replace("\n", " ")
        print(f"[{i}] (chunk_id={c['id']}, found via {'+'.join(c['sources'])})")
        print(f"    {preview}...")

    conn.close()


if __name__ == "__main__":
    main()