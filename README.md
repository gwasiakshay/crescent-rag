# crescent-rag

AI-powered client knowledge assistant for Crescent Group. Built as an internal demo for the Pachranga account.

## What it does

Takes a plain English question, searches a vector database of client account data, and returns a synthesised answer with sources.

## Stack

- **FastAPI** — `/ask` endpoint
- **Supabase pgvector** — vector database
- **OpenRouter** — embeddings (text-embedding-3-small) + synthesis (Gemini Flash)

## Setup

1. Clone the repo
2. Install dependencies: `pip3 install -r requirements.txt`
3. Create a `.env` file:
   SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_service_role_key
OPENROUTER_KEY=your_openrouter_key

4. Run locally: `python3 -m uvicorn main:app --reload`

## Endpoints

`POST /ask` — takes `{ "question": string }`, returns `{ "answer": string, "sources": [...] }`

`GET /health` — returns `{ "status": "ok" }`

## Built by

Akshay Gwasikoti · Crescent Group
