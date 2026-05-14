import os
from typing import Optional, List

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
OPENROUTER_KEY = os.environ["OPENROUTER_KEY"]
DEMO_PASSCODE = os.getenv("DEMO_PASSCODE")

# ── Clients ───────────────────────────────────────────────────────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

openai_client = OpenAI(
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    passcode: Optional[str] = None
    history: Optional[List[Message]] = []


class PasscodeRequest(BaseModel):
    passcode: Optional[str] = None


# ── Demo access control ───────────────────────────────────────────────────────
def verify_demo_passcode(
    body_passcode: Optional[str],
    header_passcode: Optional[str]
) -> None:

    if not DEMO_PASSCODE:
        return

    provided_passcode = header_passcode or body_passcode

    if provided_passcode != DEMO_PASSCODE:
        raise HTTPException(
            status_code=401,
            detail="Invalid demo passcode"
        )


# ── /verify-passcode endpoint ────────────────────────────────────────────────
@app.post("/verify-passcode")
async def verify_passcode(
    req: PasscodeRequest,
    x_demo_passcode: Optional[str] = Header(default=None)
):

    verify_demo_passcode(req.passcode, x_demo_passcode)

    return {"valid": True}


# ── Query expansion ───────────────────────────────────────────────────────────
QUERY_SYNONYMS = {
    "roi": "roas return on ad spend marketing performance",
    "renewal": "renew reactivate subscription expiry",
    "blockers": "blocked stuck pending hold stalled cannot proceed",
    "stuck": "blocked stalled cannot proceed on hold",
    "stalled": "blocked stuck pending hold",
    "hold": "blocked stalled waiting cannot proceed",
    "campaign": "advertising ads performance",
    "organic": "organic sales non paid revenue",
    "paid": "paid ads advertising spend",
}


# ── Analytical intent detection ───────────────────────────────────────────────
ANALYTICAL_KEYWORDS = [
    "improve",
    "optimize",
    "strategy",
    "recommend",
    "suggest",
    "increase",
    "reduce",
    "why",
    "issue",
    "problem",
    "risk",
    "better",
    "fix",
    "how"
]


# ── /ask endpoint ─────────────────────────────────────────────────────────────
@app.post("/ask")
async def ask(
    req: AskRequest,
    x_demo_passcode: Optional[str] = Header(default=None)
):

    verify_demo_passcode(req.passcode, x_demo_passcode)

    # ── 1. Build conversational retrieval context ────────────────────────────
    conversation_context = ""

    recent_history = req.history[-4:] if req.history else []

    for msg in recent_history:
        conversation_context += f"{msg.role}: {msg.content}\n"

    conversation_context += f"user: {req.question}"

    # ── 2. Semantic expansion ────────────────────────────────────────────────
    question = conversation_context.lower()

    for k, v in QUERY_SYNONYMS.items():
        if k in question:
            question += " " + v

    # ── 3. Detect analytical intent ──────────────────────────────────────────
    is_analytical = any(
        keyword in question
        for keyword in ANALYTICAL_KEYWORDS
    )

    # ── 4. Generate embedding ────────────────────────────────────────────────
    embed_response = openai_client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=question
    )

    query_embedding = embed_response.data[0].embedding

    # ── 5. Vector search ─────────────────────────────────────────────────────
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.28,
            "match_count": 8
        }
    ).execute()

    chunks = result.data or []

    # ── 6. No-results guard ──────────────────────────────────────────────────
    if not chunks:
        return {
            "answer": (
                "I don't have enough information in the current "
                "data to answer that question."
            ),
            "sources": []
        }

    # ── 7. Build structured retrieval context ────────────────────────────────
    context = "\n\n".join([
        (
            f"[Document {i+1}]\n"
            f"Platform: {c.get('platform', '')}\n"
            f"Job: {c.get('job', '')}\n"
            f"Status: {c.get('status', '')}\n"
            f"Content:\n{c.get('content', '')}"
        )
        for i, c in enumerate(chunks)
    ])

    # ── 8. Build dynamic system prompt ───────────────────────────────────────
    system_prompt = (
        "You are an internal operations memory assistant for Crescent Group.\n\n"

        "Rules:\n"
        "- Answer ONLY using retrieved context.\n"
        "- Give complete professional sentences.\n"
        "- Never return incomplete thoughts.\n"
        "- Mention platform names, campaign details, statuses, blockers, and operational insights when relevant.\n"
        "- Treat short follow-up questions as continuation of previous discussion.\n"
        "- If information is incomplete, clearly explain what is known.\n"
        "- If information is unavailable, say so clearly.\n"
        "- Do not invent fake data, metrics, or campaign performance.\n"
    )

    if is_analytical:
        system_prompt += (
            "\nAdditional behavior:\n"
            "- The user is asking for analysis or recommendations.\n"
            "- You may provide grounded operational suggestions and reasoning.\n"
            "- Base recommendations ONLY on retrieved context.\n"
            "- Explain WHY a recommendation could help.\n"
            "- Clearly distinguish observations from recommendations.\n"
        )
    else:
        system_prompt += (
            "\nAdditional behavior:\n"
            "- Focus on factual retrieval and operational summaries only.\n"
        )

    # ── 9. Build messages array ──────────────────────────────────────────────
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Add recent history
    for msg in recent_history:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })

    # Add retrieval context
    messages.append({
        "role": "user",
        "content": (
            f"Retrieved Context:\n{context}\n\n"
            f"Current Question:\n{req.question}"
        )
    })

    # ── 10. Generate response ────────────────────────────────────────────────
    synthesis = openai_client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=messages
    )

    answer = synthesis.choices[0].message.content

    # ── 11. Return sources ───────────────────────────────────────────────────
    sources = [
        {
            "platform": c.get("platform", ""),
            "job": c.get("job", ""),
            "status": c.get("status", ""),
            "similarity": round(c.get("similarity", 0), 3)
        }
        for c in chunks
    ]

    return {
        "answer": answer,
        "sources": sources
    }


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}
