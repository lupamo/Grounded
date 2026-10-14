"""Fast APi Server"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

from retrieve import retrieve, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME
from generate import generate_answer

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
	print("Loading embedding model...")
	state["embed_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
	print("loader reranker....")
	state["reranker"] = CrossEncoder(RERANKER_MODEL_NAME)
	print("Connecting to Database...")
	state["conn"] = psycopg.connect(DATABASE_URL)
	register_vector[state["conn"]]
	state["groq_client"] = Groq()
	print("Ready")

	yield

	state["conn"].close()

app = FastAPI(title="Grounded API", lifespan=lifespan)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:3000"],
	allow_methods=["POST"],
	allow_headers=["*"],
)


class QueryRequest(BaseModel):
	query: str

class Source(BaseModel):
	number: int
	chunk_id: int
	content: str
	found_via: list[str]
	rerank_score: float

class QueryResponse(BaseModel):
	answer: str
	sources: list[Source]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
	if not request.query.strip():
		raise HTTPException(status_code=400, detail="Query cannot be empty.")
	chunks = retrieve(request.query, state["embed_model"], state["reranker"], state["conn"])
	answer = generate_answer(request.query, chunks, state["groq_client"])

	sources = [
		Source(
			number=i,
			chunk_id=c["id"],
			content=c["content"],
			found_via=c["sources"],
			rerank_score=c["rerank_score"],
		)
		for i, c in enumerate(chunks, 1)
	]

	return QueryResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
	return {"Status": "ok"}

