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
    mode: Optional[str] = "auto"
    client: Optional[str] = "Pachranga"


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


# ── Query expansion ───────────────────────────────────────────────────────────
QUERY_SYNONYMS = {
    "roi":      "roas return on ad spend marketing performance",
    "renewal":  "renew reactivate subscription expiry",
    "blockers": "blocked stuck pending hold stalled cannot proceed",
    "stuck":    "blocked stalled cannot proceed on hold",
    "stalled":  "blocked stuck pending hold",
    "hold":     "blocked stalled waiting cannot proceed",
    "campaign": "advertising ads performance",
    "organic":  "organic sales non paid revenue",
    "paid":     "paid ads advertising spend",
}


# ── Auto intent router ────────────────────────────────────────────────────────
INTENT_RULES = {
    "analyst": [
        "brief", "briefing", "overview", "summary", "full picture",
        "everything", "all of it", "what's going on", "overall",
        "account status", "give me a rundown", "full report",
        "what do you know", "tell me about", "analyse", "analyze",
        "improve", "what needs to be done", "what should we do",
        "what can we do", "how do we", "next steps", "recommendations",
        "what to do", "action", "suggest", "fix", "better",
        "any blockers", "what's blocked", "what is blocked", "blockers",
        "what's pending", "what is pending", "pending items",
        "plan", "roadmap", "strategy", "strategic", "devise", "implement",
        "steps", "how do we execute", "2 day", "week plan", "phase",
        "prioritize", "prioritise", "focus on", "what should i",
        "timeline", "history", "what was done", "what have we done",
        "executed", "completed", "what happened", "when did",
        "chronological", "sequence", "log",
        "tasks pending", "pending tasks", "to do", "todo",
        "outstanding", "remaining", "what's remaining",
        "what all", "what have i done", "what did i do",
        "work done", "done so far",
        "calculate", "compute", "what is the formula",
        "how much is", "work out", "figure out",
        "what would", "if spend is", "if revenue is",
        "deadline", "due date", "when is", "when do i need",
        "latest deadline", "upcoming deadline", "what's due",
        "what should be done", "how to improve", "how can we improve",
        "what should i do to improve",
    ],
    "operations": [
        "stuck", "renewal", "renew", "follow up", "followup",
        "no response", "waiting", "escalate", "urgent", "overdue",
        "approval", "onboarding", "shipment", "out of stock", "inventory",
    ],
    "performance": [
        "roas", "roi", "spend", "budget", "optimise", "optimize",
        "efficiency", "cost", "burn rate", "conversion", "acos",
        "ad spend", "realloc", "underperform",
    ],
    "marketing": [
        "keyword", "campaign", "search term", "targeting", "asin",
        "bid", "creative", "content", "listing", "brand",
        "display", "remarketing",
    ],
    "memory": [],
}


def detect_intent(question: str) -> str:
    """
    Rule-based intent router.
    Checks question against keyword lists in priority order.
    Falls back to 'memory' if nothing matches.
    """
    q = question.lower()
    for mode in ["analyst", "operations", "performance", "marketing"]:
        for keyword in INTENT_RULES[mode]:
            if keyword in q:
                return mode
    return "memory"


# ── Mode-based system prompts ─────────────────────────────────────────────────
MODE_PROMPTS = {
    "memory": (
        "You are an internal operations memory assistant for Crescent Group.\n"
        "Focus on factual retrieval and operational summaries only.\n"
        "Answer ONLY using retrieved context. Do not invent facts or metrics.\n"
    ),
    "marketing": (
        "You are a marketing analyst for Crescent Group.\n"
        "Analyze campaign performance, ROAS, keywords, and spend.\n"
        "Be comprehensive — include campaign names, budgets, spend, "
        "targeting types, and optimization cadence when available.\n"
        "Provide grounded recommendations based ONLY on retrieved context.\n"
        "Clearly distinguish observations from recommendations.\n"
        "Explain WHY a recommendation could help based on the data.\n"
    ),
    "operations": (
        "You are an operations assistant for Crescent Group.\n"
        "Focus on blockers, renewals, pending approvals, and escalations.\n"
        "Surface what needs immediate attention. Be concise and action-oriented.\n"
        "Prioritise urgency — flag anything overdue or at risk.\n"
    ),
    "performance": (
        "You are a performance optimization assistant for Crescent Group.\n"
        "Focus on budget efficiency, ROAS improvement, and spend optimization.\n"
        "Identify underperforming campaigns and suggest reallocation based ONLY on retrieved context.\n"
        "Always ground recommendations in the data — never invent metrics.\n"
    ),
    "analyst": (
        "You are a senior account analyst for Crescent Group.\n"
        "You have been given the COMPLETE JSR data for this client — every row, every platform, every status.\n"
        "Your job is to reason across ALL of it, not just answer the question literally.\n\n"
        "Response structure (always follow this):\n"
        "1. Direct Answer — answer the user's specific question first, clearly and precisely. "
        "IMPORTANT: match this section exactly to what was asked. "
        "If the question is about improving ROI/ROAS, give specific data-backed improvements — not a list of blockers. "
        "If the question is about a deadline, state the deadline. "
        "If the question is about what was done, summarise the work. "
        "Never copy-paste the blockers list into the Direct Answer section.\n"
        "2. 📊 Performance Summary — key metrics (ROAS, revenue, orders, spend) if available.\n"
        "3. ✅ What's Working — active campaigns, live platforms, ongoing tasks running well.\n"
        "4. 🚨 Blockers & At-Risk Items — stalled work, no-response follow-ups, pending confirmations, out-of-stock issues.\n"
        "5. ⚡ Immediate Actions Needed — specific things that need to happen now. "
        "Only mention an owner if explicitly stated in the JSR data. Never write placeholder text like [OWNER NEEDED].\n\n"
        "Mathematical calculations:\n"
        "- If the user asks for any calculation, perform it directly using numbers from the JSR data.\n"
        "- Always show the formula, the numbers used, and the result clearly.\n"
        "- Example format: ROAS = Revenue ÷ Ad Spend = ₹5,98,496 ÷ ₹2,13,305 = 2.80\n"
        "- If a standard metric cannot be calculated (e.g. organic ROAS where spend = 0), explain why "
        "and offer an alternative calculation instead (e.g. organic revenue as % of total revenue).\n"
        "- Never skip a calculation if the numbers are available in the data.\n\n"
        "Rules:\n"
        "- Always surface blockers even if the user didn't ask about them.\n"
        "- Be specific — name the platform, the job, the issue.\n"
        "- Never invent data. Only use what's in the JSR.\n"
        "- Think like a smart analyst who has read the full file, not a search engine.\n"
    ),
}

# Shared rules appended to non-analyst mode prompts
SHARED_RULES = (
    "\nRules:\n"
    "- Answer ONLY using the retrieved context.\n"
    "- Give complete and professional sentences.\n"
    "- Never return incomplete sentences.\n"
    "- Mention platform names, campaign details, statuses, and blockers when relevant.\n"
    "- Treat short follow-up questions as continuation of the previous discussion.\n"
    "- If information is incomplete, clearly explain what is known.\n"
    "- If information is unavailable, say so clearly.\n"
    "- Do not invent facts, metrics, or campaign performance.\n"
)


# ── Analyst mode: fetch ALL documents for a client ────────────────────────────
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
            .limit(4)
            .execute()
        )
        recent_history = [
            Message(role=m["role"], content=m["content"])
            for m in reversed(db_msgs.data or [])
        ]
    else:
        conv_row = supabase.table("conversations").insert({}).execute()
        conversation_id = conv_row.data[0]["id"]
        recent_history = req.history[-4:] if req.history else []

    # ── 2. Resolve mode ───────────────────────────────────────────────────────
    if req.mode == "auto" or req.mode not in MODE_PROMPTS:
        resolved_mode = detect_intent(req.question)
    else:
        resolved_mode = req.mode

    # ── 3. ANALYST MODE — full context path ───────────────────────────────────
    if resolved_mode == "analyst":
        full_context = fetch_all_client_docs(req.client or "Pachranga")

        if not full_context:
            return {
                "answer": "No data found for this client in the database.",
                "sources": [],
                "conversation_id": conversation_id,
                "mode_used": resolved_mode,
            }

        messages = [{"role": "system", "content": MODE_PROMPTS["analyst"]}]
        for msg in recent_history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({
            "role": "user",
            "content": (
                f"Complete JSR Data for {req.client or 'Pachranga'}:\n\n"
                f"{full_context}\n\n"
                f"Question: {req.question}"
            )
        })

        synthesis = openai_client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=messages
        )
        answer = synthesis.choices[0].message.content

        supabase.table("messages").insert([
            {"conversation_id": conversation_id, "role": "user", "content": req.question},
            {"conversation_id": conversation_id, "role": "assistant", "content": answer, "sources": []},
        ]).execute()

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

        return {
            "answer": answer,
            "sources": [],
            "conversation_id": conversation_id,
            "mode_used": resolved_mode,
        }

    # ── 4. STANDARD MODES — vector search path ────────────────────────────────

    # Build conversational retrieval context
    conversation_context = ""
    for msg in recent_history:
        conversation_context += f"{msg.role}: {msg.content}\n"
    conversation_context += f"user: {req.question}"

    # Semantic expansion
    question = conversation_context.lower()
    for k, v in QUERY_SYNONYMS.items():
        if k in question:
            question += " " + v

    # Generate embedding
    embed_response = openai_client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=question
    )
    query_embedding = embed_response.data[0].embedding

    # Vector search
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.28,
            "match_count": 8
        }
    ).execute()

    chunks = result.data or []

    # No-results guard
    if not chunks:
        return {
            "answer": "I don't have enough information in the current data to answer that question.",
            "sources": [],
            "conversation_id": conversation_id,
            "mode_used": resolved_mode,
        }

    # Build structured retrieval context
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

    # Build mode-based system prompt
    system_prompt = MODE_PROMPTS[resolved_mode] + SHARED_RULES

    # Build messages array
    messages = [{"role": "system", "content": system_prompt}]
    for msg in recent_history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({
        "role": "user",
        "content": f"Retrieved Context:\n{context}\n\nCurrent Question:\n{req.question}"
    })

    # Generate response
    synthesis = openai_client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=messages
    )
    answer = synthesis.choices[0].message.content

    # Build sources
    sources = [
        {
            "platform": c.get("platform", ""),
            "job": c.get("job", ""),
            "status": c.get("status", ""),
            "similarity": round(c.get("similarity", 0), 3)
        }
        for c in chunks
    ]

    # Persist messages & update conversation
    supabase.table("messages").insert([
        {"conversation_id": conversation_id, "role": "user", "content": req.question},
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
        conv_update["title"] = req.question[:80]
    supabase.table("conversations").update(conv_update).eq("id", conversation_id).execute()

    return {
        "answer": answer,
        "sources": sources,
        "conversation_id": conversation_id,
        "mode_used": resolved_mode,
    }


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}
