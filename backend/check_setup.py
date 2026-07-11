"""
confirms Postgres is reachable and pgvector is enabled
before running the app, run `python backend/check_setup.py` to check if the database is reachable and pgvector is enabled.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
	print("DATABASE_URL is not set in the environment variables.")
	sys.exit(1)

try:
	conn = psycopg.connect(DATABASE_URL)
except Exception as e:
	print(f"Could not connect to postgres: {e}")
	print("Check if container is running. Try: docker compose up -d")
	sys.exit(1)

with conn.cursor() as cur:
	cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
	row = cur.fetchone()
	if row:
		print("pgvector extension is enabled.")
	else:
		print("pgvector extension is NOT enabled. Check init.sql ran correctly.")
		sys.exit(1)
	
	cur.execute("SELECT version();")
	print(f"Connected to: {cur.fetchone()[0]}")

conn.close()
print("n\Setup check passed.")
