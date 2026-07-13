# A RAG question and answering system for PDFs
You ingest PDF and it extracts text and splits it into chunks, creating local embedings for each chunk and stores them in PostgreSQL with pgvector.
When asked a question the system retrieves candidate chunks using both:
	- semantic/vector similarity search
	- PostgreSQL full text key word search
Merges and reranks those results with local cross-encoder model.

### Features
- Hybrid retrieval: pgvector semantic search + Postgress full-text (BM25-style) search, merged
- Reranking step (cross-encoder or Cohere rerank) before generation
- Inline citations - every claim links to the exacr source chunk, highlighted.
- Live eval panel: faithfulness score (does the answer actially follow from retrived chunks?) and retrieval precision, computed per-query with LLM as  judge right in the UI.

## Stack
- Embeddings: BAAI/bge-small-en-v1.5 via sentence-transformer, local, freee
-  Reranking: cross-encoder/ms-macro-MiniLM-L-6-v2, local free
- Generation: Groq(llama-3.3-70b-versatile)
- Retrival: pgvector (semantic) + Postgress full-text(keyword), merged, no cost just postgress

## Enviroment Setup
1. Start Postgress (with Pgvector) via Docker:
 ```bash
	docker compose up -d
 ```
 This will pull the pgvector/pgvector:pg16 image and initialize the database, running init.sql automatically to enable the vector extension.

 2. Set up the Python backend enviroment:
 ```bash
	cd backend
	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirement.txt
	cp .env.example .env
 ``` 
The .env defaults already match docker-compose.yml credentials, so you don't need to edit DATABASE_URL. You'll need to add your real GROQ_API_KEY. 
sentence-transformers will download two models on first use (an embedding model and a reranking model) both run locally on your CPU, no API or cost inolved

3. Verify everything is wired up correctly:
```bash
	python check_setup.py
```
```expected-output
	pgvector extension is enabled.
	Connected to: PostgreSQL 16.x ....
	Setup check passed. Ready for step 2 (schema. design)
```