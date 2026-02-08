Backend (FastAPI)
==================

This folder contains a minimal FastAPI scaffold for the Chain-Reaction backend.

Setup (macOS / Linux):

1. python -m venv .venv
2. source .venv/bin/activate
3. pip install -r requirements.txt
4. uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

Notes:
- Implement routes for ingestion, ETL orchestration, model serving, and webhooks here.
- Use `motor` to connect to MongoDB Atlas and add geospatial indexes as needed.
