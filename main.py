import os
from typing import Optional, List
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv
load_dotenv()
import httpx

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
OPENROUTER_KEY = os.environ["OPENROUTER_KEY"]
DEMO_PASSCODE  = os.getenv("DEMO_PASSCODE")

# ── Clients ───────────────────────────────────────────────────────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

openai_client = OpenAI(
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class AskRequest(BaseModel):
    question: str
    passcode: Optional[str] = None
    history: Optional[List[Message]] = []

class PasscodeRequest(BaseModel):
    passcode: Optional[str] = None


def verify_demo_passcode(body_passcode: Optional[str], header_passcode: Optional[str]) -> None:
    if not DEMO_PASSCODE:
        return
    provided_passcode = header_passcode or body_passcode
    if provided_passcode != DEMO_PASSCODE:
        raise HTTPException(status_code=401, detail="Invalid demo passcode")

# ── /verify-passcode endpoint ─────────────────────────────────────────────────
@app.post("/verify-passcode")
async def verify_passcode(req: PasscodeRequest, x_demo_passcode: Optional[str] = Header(default=None)):
    verify_demo_passcode(req.passcode, x_demo_passcode)
    return {"valid": True}

# ── /ask endpoint ─────────────────────────────────────────────────────────────
@app.post("/ask")
async def ask(req: AskRequest, x_demo_passcode: Optional[str] = Header(default=None)):
    verify_demo_passcode(req.passcode, x_demo_passcode)

    # 1. Build richer embedding query for short/vague questions
    if len(req.question.split()) <= 4 and req.history:
        recent_user_msgs = [m.content for m in req.history[-4:] if m.role == "user"]
        question_to_embed = " ".join(recent_user_msgs + [req.question])
    else:
        question_to_embed = req.question

    # 2. Embed the question
    embed_response = openai_client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=question_to_embed
    )
    query_embedding = embed_response.data[0].embedding

    # 3. Vector search via Supabase RPC
    result = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_threshold": 0.35,
        "match_count": 8
    }).execute()

    chunks = result.data or []

    # 4. No-results guard — don't call Gemini if nothing relevant found
    if not chunks:
        return {
            "answer": "I don't have enough information in the current data to answer that question.",
            "sources": []
        }

    # 5. Build context for Gemini
    context = "\n\n".join([
        f"[{i+1}] Platform: {c.get('platform','')}, Job: {c.get('job','')}, "
        f"Status: {c.get('status','')}\n{c.get('content','')}"
        for i, c in enumerate(chunks)
    ])

    # 6. Sliding window — last 4 messages only
    recent_history = req.history[-4:] if len(req.history) > 4 else req.history

    # 7. Build messages array with history
    messages = [
        {
            "role": "system",
            "content": (
                "You are a media account assistant for Crescent Group. "
                "Answer questions about client campaigns using only the context provided. "
                "Be specific - include platform names, status, budget figures, and remarks from the context. "
                "Do not give one-line answers. Summarise all relevant details you find. "
                "If the user sends a short follow-up like 'organic?' or 'paid?' treat it as a continuation "
                "of the previous question and answer accordingly from the context. "
                "If the context does not contain enough information to answer, say so clearly — do not invent or assume data."
            )
        }
    ]

    for msg in recent_history:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {req.question}"
    })

    # 8. Synthesise with Gemini Flash
    synthesis = openai_client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=messages
    )

    answer = synthesis.choices[0].message.content

    # 9. Return answer + sources
    sources = [
        {
            "platform": c.get("platform", ""),
            "job": c.get("job", ""),
            "status": c.get("status", ""),
            "similarity": round(c.get("similarity", 0), 3)
        }
        for c in chunks
    ]

    return {"answer": answer, "sources": sources}


@app.get("/health")
async def health():
    return {"status": "ok"}
