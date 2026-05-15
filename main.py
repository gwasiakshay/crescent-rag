import os
from typing import Optional, List, Any

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

# ── Request / response schemas ────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    passcode: Optional[str] = None
    history: Optional[List[Message]] = []
    conversation_id: Optional[str] = None


class PasscodeRequest(BaseModel):
    passcode: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: Optional[List[Any]]
    created_at: str


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


# ── /conversations endpoints ──────────────────────────────────────────────────
@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    x_demo_passcode: Optional[str] = Header(default=None)
):
    verify_demo_passcode(None, x_demo_passcode)

    row = supabase.table("conversations").insert({}).execute()
    data = row.data[0]
    return ConversationResponse(
        id=data["id"],
        title=data.get("title"),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


@app.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    x_demo_passcode: Optional[str] = Header(default=None)
):
    verify_demo_passcode(None, x_demo_passcode)

    rows = (
        supabase.table("conversations")
        .select("*")
        .order("updated_at", desc=True)
        .execute()
    )
    return [
        ConversationResponse(
            id=r["id"],
            title=r.get("title"),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows.data
    ]


@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conversation_id: str,
    x_demo_passcode: Optional[str] = Header(default=None)
):
    verify_demo_passcode(None, x_demo_passcode)

    rows = (
        supabase.table("messages")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("created_at")
        .execute()
    )
    return [
        MessageResponse(
            id=r["id"],
            role=r["role"],
            content=r["content"],
            sources=r.get("sources"),
            created_at=r["created_at"],
        )
        for r in rows.data
    ]


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

    # ── 1. Resolve conversation & history ────────────────────────────────────
    if req.conversation_id:
        conversation_id = req.conversation_id
        # Load last 4 messages in correct chronological order
        db_msgs = (
            supabase.table("messages")
            .select("role,content")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=True)  # newest first
            .limit(4)
            .execute()
        )
        # Reverse so oldest is first (correct conversation order)
        recent_history = [
            Message(role=m["role"], content=m["content"])
            for m in reversed(db_msgs.data or [])
        ]
    else:
        # Auto-create a new conversation (backwards-compatible)
        conv_row = supabase.table("conversations").insert({}).execute()
        conversation_id = conv_row.data[0]["id"]
        recent_history = req.history[-4:] if req.history else []

    # ── 2. Build conversational retrieval context ────────────────────────────
    conversation_context = ""
    for msg in recent_history:
        conversation_context += f"{msg.role}: {msg.content}\n"
    conversation_context += f"user: {req.question}"

    # ── 3. Semantic expansion ────────────────────────────────────────────────
    question = conversation_context.lower()
    for k, v in QUERY_SYNONYMS.items():
        if k in question:
            question += " " + v

    # ── 4. Detect analytical intent ──────────────────────────────────────────
    is_analytical = any(
        keyword in question
        for keyword in ANALYTICAL_KEYWORDS
    )

    # ── 5. Generate embedding ────────────────────────────────────────────────
    embed_response = openai_client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=question
    )
    query_embedding = embed_response.data[0].embedding

    # ── 6. Vector search ─────────────────────────────────────────────────────
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.28,
            "match_count": 8
        }
    ).execute()

    chunks = result.data or []

    # ── 7. No-results guard ──────────────────────────────────────────────────
    if not chunks:
        return {
            "answer": (
                "I don't have enough information in the current "
                "data to answer that question."
            ),
            "sources": [],
            "conversation_id": conversation_id,
        }

    # ── 8. Build structured retrieval context ────────────────────────────────
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

    # ── 9. Build dynamic system prompt ───────────────────────────────────────
    system_prompt = (
        "You are an internal operations memory assistant for Crescent Group.\n\n"

        "Rules:\n"
        "- Answer ONLY using the retrieved context.\n"
        "- Give complete and professional sentences.\n"
        "- Never return incomplete sentences.\n"
        "- Mention platform names, campaign details, statuses, and blockers when relevant.\n"
        "- Treat short follow-up questions as continuation of the previous discussion.\n"
        "- If information is incomplete, clearly explain what is known.\n"
        "- If information is unavailable, say so clearly.\n"
        "- Do not invent facts, metrics, or campaign performance.\n"
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

    # ── 10. Build messages array ──────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]

    # Add recent history for conversational context
    for msg in recent_history:
        messages.append({"role": msg.role, "content": msg.content})

    # Add retrieval context + current question
    messages.append({
        "role": "user",
        "content": (
            f"Retrieved Context:\n{context}\n\n"
            f"Current Question:\n{req.question}"
        )
    })

    # ── 11. Generate response ─────────────────────────────────────────────────
    synthesis = openai_client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=messages
    )

    answer = synthesis.choices[0].message.content

    # ── 12. Build sources ─────────────────────────────────────────────────────
    sources = [
        {
            "platform": c.get("platform", ""),
            "job": c.get("job", ""),
            "status": c.get("status", ""),
            "similarity": round(c.get("similarity", 0), 3)
        }
        for c in chunks
    ]

    # ── 13. Persist messages & update conversation ────────────────────────────
    supabase.table("messages").insert([
        {"conversation_id": conversation_id, "role": "user", "content": req.question},
        {"conversation_id": conversation_id, "role": "assistant", "content": answer, "sources": sources},
    ]).execute()

    # Set title from first question, bump updated_at
    conv_update: dict = {"updated_at": "now()"}
    existing = (
        supabase.table("conversations")
        .select("title")
        .eq("id", conversation_id)
        .execute()
    )
    if existing.data and existing.data[0].get("title") is None:
        conv_update["title"] = req.question[:80]
    supabase.table("conversations").update(conv_update).eq("id", conversation_id).execute()

    # ── 14. Return ────────────────────────────────────────────────────────────
    return {
        "answer": answer,
        "sources": sources,
        "conversation_id": conversation_id,
    }


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}
