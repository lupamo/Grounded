"""
	applies schema.sql ro create the documents and chunks tables
	run:
		python backend/apply_schema.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

if not DATABASE_URL:
	print("DATABASE_URL is not set in the environment variables.")
	sys.exit(1)
schema_sql = SCHEMA_PATH.read_text()

try:
	conn = psycopg.connect(DATABASE_URL, autocommit=True)
except Exception as e:
	print(f"Could not connect to postgres: {e}")
	sys.exit(1)
with conn.cursor() as cur:
	cur.execute(schema_sql)

with conn.cursor() as cur:
	cur.execute(
	"""
	SELECT table_name FROM information_schema.tables
	WHERE table_schema = 'public' AND table_name IN ('documents', 'chunks')
	ORDER BY table_name;
	"""
	)
	tables = [row[0] for row in cur.fetchall()]

conn.close()
if set(tables) == {"chunks", "documents"}:
	print("Schema applied successfully. Tables 'documents' and 'chunks' created.")
	print("Ready for ingestion (chunking + embedding) real documents")
else:
	print(f"Something went wrong. only found tables: {tables}.")
	sys.exit(1)
