import os
from typing import Optional, List, Any

from datetime import date
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

# ── Available clients ─────────────────────────────────────────────────────────
AVAILABLE_CLIENTS = [
    "Pachranga",
    "Crespack",
    "Jaquar",
    "Jaquar International",
    "Jaquar GMB",
    "Jaquar Lighting",
    "British Paint",
    "Essco",
    "Canon",
    "TCI Logistics",
    "Artize",
    "TRE",
    "FX10",
    "WATI",
    "Maspar",
    "Crescent Lead Campaign",
    "Snack Factory",
    "Saraswati",
]

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
    mode: Optional[str] = "auto"
    client: Optional[str] = None


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
        raise HTTPException(status_code=401, detail="Invalid demo passcode")


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


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    x_demo_passcode: Optional[str] = Header(default=None)
):
    verify_demo_passcode(None, x_demo_passcode)
    supabase.table("conversations").delete().eq("id", conversation_id).execute()
    return {"deleted": True}


@app.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    body: dict,
    x_demo_passcode: Optional[str] = Header(default=None)
):
    verify_demo_passcode(None, x_demo_passcode)
    supabase.table("conversations").update(
        {"title": body.get("title")}
    ).eq("id", conversation_id).execute()
    return {"updated": True}


# ── Clarification system ──────────────────────────────────────────────────────
AMBIGUOUS_KEYWORDS = [
    "roi", "roas", "revenue", "performance", "summary", "overview",
    "status", "what's going on", "campaigns", "blockers", "pending",
    "tasks", "what needs", "improve", "brief", "briefing", "tell me",
    "what have", "what was done", "what all", "what should",
    "deadline", "what's due", "remaining", "current", "budget",
    "what are", "how is", "how are", "give me",
]

def detect_client_from_question(question: str):
    q = question.lower()
    for c in AVAILABLE_CLIENTS:
        if c.lower() in q:
            return c
    return None

def needs_clarification(question: str, client):
    if len(AVAILABLE_CLIENTS) <= 1:
        return None
    if client and client in AVAILABLE_CLIENTS:
        return None
    if detect_client_from_question(question):
        return None
    q = question.lower()
    if any(kw in q for kw in AMBIGUOUS_KEYWORDS):
        client_options = ", ".join(AVAILABLE_CLIENTS[:-1]) + f", or {AVAILABLE_CLIENTS[-1]}"
        return (
            f"Which account are you asking about — {client_options}? "
            f"Or would you like a combined overview of all accounts?"
        )
    return None


# ── Full context fetch ────────────────────────────────────────────────────────
def fetch_all_client_docs(client: str) -> str:
    """Fetch every document for a client and build a full context string."""
    rows = (
        supabase.table("documents")
        .select("content, platform, job, status")
        .eq("client", client)
        .execute()
    )
    if not rows.data:
        return ""
    parts = []
    for i, r in enumerate(rows.data):
        parts.append(
            f"[Row {i+1}]\n"
            f"Platform: {r.get('platform', '')}\n"
            f"Job: {r.get('job', '')}\n"
            f"Status: {r.get('status', '')}\n"
            f"Content:\n{r.get('content', '')}"
        )
    return "\n\n".join(parts)


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    f"Today's date is {date.today().strftime('%B %d, %Y')}. "
    "Use this to assess deadlines — flag any that have already passed and state how many days away upcoming ones are.\n\n"
    "You are a senior account analyst and operations assistant for Crescent Group, "
    "a media and marketing agency.\n"
    "You have been given the COMPLETE JSR (Job Sheet Report) data for the client — "
    "every row, every platform, every task, every status.\n\n"

    "Your job is to reason across ALL of it like a smart analyst who has read the "
    "full file — not a search engine retrieving chunks.\n\n"

    "Adapt your response structure to what was actually asked:\n\n"

    "For FACTUAL questions (what is the roi, what are the tasks, give me an overview, "
    "any blockers, what's pending, current tasks):\n"
    "1. Direct Answer — precise and complete.\n"
    "2. 📊 Performance Summary — key metrics if available.\n"
    "3. ✅ What's Working\n"
    "4. 🚨 Blockers & At-Risk Items\n"
    "5. ⚡ Immediate Actions Needed — no placeholder owners.\n\n"

    "For STRATEGIC questions (how to improve, implementation plan, strategy, "
    "roadmap, next steps, what should we do):\n"
    "1. Direct Answer — 2-3 sentence strategic summary.\n"
    "2. Detailed plan with numbered sections and specific actions.\n"
    "Skip Performance Summary and Blockers — user already has that context.\n\n"

    "For CALCULATION questions (calculate roi, what is the formula, compute):\n"
    "1. Show formula + numbers + result clearly.\n"
    "   Example: ROAS = Revenue ÷ Ad Spend = ₹54,757 ÷ ₹96,812 = 0.57\n"
    "2. If a metric can't be calculated (e.g. organic ROAS, spend = 0), explain why "
    "and offer an alternative (e.g. organic % of total revenue).\n"
    "Skip the full 5-section structure for pure calculation questions.\n\n"

    "For SHORT FOLLOW-UP questions in an ongoing conversation:\n"
    "Answer directly without repeating context already given.\n\n"

    "Mathematical rules:\n"
    "- Always perform calculations when numbers are available.\n"
    "- Never skip a calculation if the data supports it.\n\n"

    "General rules:\n"
    "- Answer ONLY using the provided JSR data. Never invent facts or metrics.\n"
    "- Be specific — name the platform, job, status, issue.\n"
    "- Surface blockers proactively even when not asked.\n"
    "- Only mention task owners if explicitly stated in the data.\n"
    "- Never write placeholder text like [OWNER NEEDED].\n"
)

# ── Query expansion (kept for vector search fallback) ─────────────────────────
QUERY_SYNONYMS = {
    "roi":      "roas return on ad spend marketing performance",
    "renewal":  "renew reactivate subscription expiry",
    "blockers": "blocked stuck pending hold stalled cannot proceed",
    "stuck":    "blocked stalled cannot proceed on hold",
    "campaign": "advertising ads performance",
    "organic":  "organic sales non paid revenue",
    "paid":     "paid ads advertising spend",
}


# ── Persist & update conversation ─────────────────────────────────────────────
def persist(conversation_id: str, question: str, answer: str, sources: list):
    supabase.table("messages").insert([
        {"conversation_id": conversation_id, "role": "user", "content": question},
        {"conversation_id": conversation_id, "role": "assistant", "content": answer, "sources": sources},
    ]).execute()

    conv_update: dict = {"updated_at": "now()"}
    existing = (
        supabase.table("conversations")
        .select("title")
        .eq("id", conversation_id)
        .execute()
    )
    if existing.data and existing.data[0].get("title") is None:
        conv_update["title"] = question[:80]
    supabase.table("conversations").update(conv_update).eq("id", conversation_id).execute()


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
        db_msgs = (
            supabase.table("messages")
            .select("role,content")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=True)
            .limit(6)
            .execute()
        )
        recent_history = [
            Message(role=m["role"], content=m["content"])
            for m in reversed(db_msgs.data or [])
        ]
    else:
        conv_row = supabase.table("conversations").insert({}).execute()
        conversation_id = conv_row.data[0]["id"]
        recent_history = req.history[-6:] if req.history else []

    # ── 2. Resolve client ─────────────────────────────────────────────────────
    client_from_question = detect_client_from_question(req.question)
    resolved_client = client_from_question or req.client or None

    # ── 3. Clarification check ────────────────────────────────────────────────
    clarification = needs_clarification(req.question, resolved_client)
    if clarification:
        persist(conversation_id, req.question, clarification, [])
        return {
            "answer": clarification,
            "sources": [],
            "conversation_id": conversation_id,
            "mode_used": "clarification",
            "needs_clarification": True,
            "available_clients": AVAILABLE_CLIENTS,
        }

    # Default client if still unresolved
    if not resolved_client:
        resolved_client = AVAILABLE_CLIENTS[0]

    # ── 4. PRIMARY PATH — full context (default for all queries) ──────────────
    full_context = fetch_all_client_docs(resolved_client)

    if full_context:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in recent_history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({
            "role": "user",
            "content": (
                f"Complete JSR data for {resolved_client}:\n\n"
                f"{full_context}\n\n"
                f"Question: {req.question}"
            )
        })

        synthesis = openai_client.chat.completions.create(
            model="anthropic/claude-haiku-4-5",
            messages=messages
        )
        answer = synthesis.choices[0].message.content

        persist(conversation_id, req.question, answer, [])

        return {
            "answer": answer,
            "sources": [],
            "conversation_id": conversation_id,
            "mode_used": "full_context",
            "client_used": resolved_client,
        }

    # ── 5. FALLBACK — vector search (when no client docs found) ───────────────
    conversation_context = ""
    for msg in recent_history:
        conversation_context += f"{msg.role}: {msg.content}\n"
    conversation_context += f"user: {req.question}"

    question_expanded = conversation_context.lower()
    for k, v in QUERY_SYNONYMS.items():
        if k in question_expanded:
            question_expanded += " " + v

    embed_response = openai_client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=question_expanded
    )
    query_embedding = embed_response.data[0].embedding

    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.28,
            "match_count": 8
        }
    ).execute()

    chunks = result.data or []

    if not chunks:
        answer = "I don't have enough information in the current data to answer that question."
        persist(conversation_id, req.question, answer, [])
        return {
            "answer": answer,
            "sources": [],
            "conversation_id": conversation_id,
            "mode_used": "fallback",
        }

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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in recent_history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({
        "role": "user",
        "content": f"Retrieved Context:\n{context}\n\nQuestion:\n{req.question}"
    })

    synthesis = openai_client.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        messages=messages
    )
    answer = synthesis.choices[0].message.content

    sources = [
        {
            "platform": c.get("platform", ""),
            "job": c.get("job", ""),
            "status": c.get("status", ""),
            "similarity": round(c.get("similarity", 0), 3)
        }
        for c in chunks
    ]

    persist(conversation_id, req.question, answer, sources)

    return {
        "answer": answer,
        "sources": sources,
        "conversation_id": conversation_id,
        "mode_used": "vector_search_fallback",
        "client_used": resolved_client,
    }


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── /clients endpoint ─────────────────────────────────────────────────────────
@app.get("/clients")
async def list_clients(
    x_demo_passcode: Optional[str] = Header(default=None)
):
    verify_demo_passcode(None, x_demo_passcode)
    return {"clients": AVAILABLE_CLIENTS}
