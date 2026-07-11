"""
Ingestion pipeline
takes a PDF, extracts text page, splits it into overlapping chunks along paragraph boundaries
embeds each chunk locally with bge-small-en-v15
Usage:
    python ingest.py /path/to/document.pdf --title "Agents (Google, 2024)"
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def extract_text_from_pdf(pdf_path: Path) -> str:
	"""
	Pulls text out of every page, joined with double newlines so page
	breaks behave like paragraph breaks for chunking purposes.
	"""
	reader = PdfReader(pdf_path)
	pages = [page.extract_text() for page in reader.pages]
	return "\n\n".join(p for p in pages if p and p.strip())

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
	"""
	splits on paragraph boundaries first, then packs paragraphs into chunks
	up to chunk_size, carrying a small overlap of trailing text into the
    next chunk. This is a simplified, framework-free version of what
    LangChain's RecursiveCharacterTextSplitter does.
	"""
	paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
	chunks = []
	current = ""
	for para in paragraphs:
		if len(current) + len(para) + 1 <= chunk_size:
			current = f"{current}\n{para}".strip()
		else:
			if current:
				chunks.append(current)
			overlap_text = current[-overlap:] if len(current) > overlap else current
			current =  f"{overlap_text}\n{para}".strip()

			while len(current) > chunk_size:
				chunks.append(current[:chunk_size])
				current = current[chunk_size - overlap:]
	if current:
		chunks.append(current)
	return chunks

def main():
	parser = argparse.ArgumentParser(description="Ingest a PDF into the Grounded knowledge base.")
	parser.add_argument("pdf_path", help="Path to the PDF file to ingest.")
	parser.add_argument("--title", required=True, help="Human-readable title for this document.")

	args = parser.parse_args()

	pdf_path = Path(args.pdf_path)
	if not pdf_path.exists():
		print(f"Error: File not found - {pdf_path}")
		sys.exit(1)

	print(f"Extracting text from {pdf_path.name}...")
	text = extract_text(pdf_path)
	print(f"Extracted {len(text)} characters.")

	print("Chunking...")
	chunks = chunk_text(text)
	print(f"Created {len(chunks)} chunks.")

	print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})... first run downloads it, be patient.")
	model = SentenceTransformer(EMBEDDING_MODEL_NAME)

	print("Embedding chunks (runs locally on CPU, no API calls)...")
	embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)

	print("Writing to database...")
	conn = psycopg.connect(DATABASE_URL, autocommit=True)
	register_vector(conn)

	with conn.cursor() as cur:
		cur.execute(
			"INSERT INTO documents (title, source_path) VALUES (%s, %s) RETURNING id;",
			(args.title, str(pdf_path)),
		)
		document_id = cur.fetchone()[0]

		for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
			cur.execute(
				"""
				INSERT INTO chunks (document_id, chunk_index, content, embedding)
				VALUES (%s, %s, %s, %s);
				""",
				(document_id, idx, chunk, embedding),
            )
	conn.close()
	print(f"\nDone. Document id={document_id}, {len(chunks)} chunks inserted.")
	print("Ready for step 4 (retrieval: hybrid search + reranking).")

if __name__ == "__main__":
    main()