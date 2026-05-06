"""
FastAPI Chatbot — RAG retrieval + Persona-aware answers
========================================================
Combines topic summaries, message chunks, and persona data to answer queries.
"""

import json
import faiss
import pickle
import os
import re
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel


app = FastAPI(title="RAG Persona Chatbot")

# ── Static files ───────────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Load pre-built data ───────────────────────────────────────────────
print("Loading RAG Index ...")
try:
    index = faiss.read_index('data_cache/rag_index.faiss')
    with open('data_cache/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    with open('data_cache/persona.json', 'r', encoding='utf-8') as f:
        persona = json.load(f)
    with open('data_cache/topics.json', 'r', encoding='utf-8') as f:
        topics_list = json.load(f)
    with open('data_cache/checkpoints.json', 'r', encoding='utf-8') as f:
        checkpoints_list = json.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loaded successfully.")
except Exception as e:
    print(f"Failed to load data: {e}.  Run data_pipeline.py + persona_builder.py first.")
    index = None
    metadata = []
    persona = {}
    topics_list = []
    checkpoints_list = []
    embedder = None


class ChatRequest(BaseModel):
    message: str


# ── Intent detection ───────────────────────────────────────────────────
PERSONA_HABITS_KW   = re.compile(r'\b(habit|sleep|food|eat|exercise|hobby|hobbi|routine|diet)\b', re.I)
PERSONA_STYLE_KW    = re.compile(r'\b(talk|speak|communication|style|write|tone|emoji|message)\b', re.I)
PERSONA_PERSON_KW   = re.compile(r'\b(person|personality|kind of|type of|who is|what is|trait|character)\b', re.I)


def format_persona_habits(p, user_key="user_1"):
    """Build a natural-language answer from persona habits."""
    u = p.get(user_key, {})
    if not u:
        return "No persona data available."
    h = u.get("habits", {})
    lines = ["**Habits detected from conversation signals:**\n"]
    lines.append(f"🛏️ **Sleep:** {', '.join(h.get('sleep', ['N/A']))}")
    lines.append(f"🍕 **Food:** {', '.join(h.get('food', ['N/A']))}")
    lines.append(f"🏃 **Exercise:** {', '.join(h.get('exercise', ['N/A']))}")
    lines.append(f"🎮 **Hobbies/Interests:** {', '.join(h.get('hobbies_interests', ['N/A']))}")
    return "\n".join(lines)


def format_persona_style(p, user_key="user_1"):
    u = p.get(user_key, {})
    if not u:
        return "No persona data available."
    s = u.get("communication_style", {})
    pt = u.get("personality_traits", {})
    lines = ["**Communication Style (based on message analysis):**\n"]
    lines.append(f"💬 **Tone:** {s.get('tone', 'N/A')}")
    lines.append(f"📏 **Message style:** {s.get('message_style', 'N/A')} (avg {s.get('avg_words_per_message', '?')} words/msg)")
    lines.append(f"❓ **Questions:** {s.get('question_ratio', 'N/A')}")
    lines.append(f"❗ **Exclamations:** {s.get('exclamation_ratio', 'N/A')}")
    lines.append(f"😊 **Emoji usage:** {s.get('emoji_usage', 'N/A')}")
    lines.append(f"\n**Personality signals:**")
    lines.append(f"😄 **Humor:** {pt.get('humor', 'N/A')}")
    lines.append(f"❤️ **Emotional:** {pt.get('emotional_expressiveness', 'N/A')}")
    lines.append(f"🎭 **Sentiment:** {pt.get('sentiment', 'N/A')}")
    return "\n".join(lines)


def format_persona_full(p, user_key="user_1"):
    u = p.get(user_key, {})
    if not u:
        return "No persona data available."
    h = u.get("habits", {})
    f_ = u.get("personal_facts", {})
    pt = u.get("personality_traits", {})
    s = u.get("communication_style", {})
    lines = ["**User Profile (extracted from conversation data):**\n"]
    lines.append(f"🧠 **Personality:** {pt.get('humor', '')} | {pt.get('emotional_expressiveness', '')} | {pt.get('sentiment', '')}")
    lines.append(f"💬 **Style:** {s.get('tone', '')} tone, {s.get('message_style', '')} messages")
    lines.append(f"🛏️ **Sleep:** {', '.join(h.get('sleep', ['N/A']))}")
    lines.append(f"🍕 **Food:** {', '.join(h.get('food', ['N/A']))}")
    lines.append(f"🏃 **Exercise:** {', '.join(h.get('exercise', ['N/A']))}")
    lines.append(f"🎮 **Interests:** {', '.join(h.get('hobbies_interests', ['N/A']))}")
    lines.append(f"👥 **Relationships:** {', '.join(f_.get('relationships', ['N/A']))}")
    lines.append(f"📍 **Locations:** {', '.join(f_.get('locations', ['N/A']))}")
    lines.append(f"📅 **Life events:** {', '.join(f_.get('events', ['N/A']))}")
    return "\n".join(lines)


def rag_retrieve(query, k=8):
    """Retrieve relevant documents from FAISS, returning both topic summaries and chunks."""
    q_emb = embedder.encode([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k=k)

    topic_results = []
    chunk_results = []
    checkpoint_results = []
    sources = []

    for score, idx in zip(D[0], I[0]):
        if float(score) < 0.35:
            continue
        if idx >= len(metadata):
            continue
        m = metadata[idx]
        if m["type"] == "topic":
            topic_results.append(m['data']['summary'])
            sources.append({
                "type": "topic",
                "topic_id": m['data']['topic_id'],
                "msgs": f"{m['data']['start_msg']}–{m['data']['end_msg']}",
                "keywords": m['data'].get('keywords', []),
                "preview": m['data']['summary'][:120] + "…",
                "score": round(float(score), 3),
            })
        elif m["type"] == "checkpoint":
            checkpoint_results.append(m['data']['summary'])
            sources.append({
                "type": "checkpoint",
                "checkpoint_id": m['data']['checkpoint_id'],
                "msgs": f"{m['data']['start_msg']}–{m['data']['end_msg']}",
                "preview": m['data']['summary'][:120] + "…",
                "score": round(float(score), 3),
            })
        elif m["type"] == "chunk":
            chunk_results.append(m['data']['text'])
            sources.append({
                "type": "chunk",
                "msgs": f"{m['data']['start_idx']}–{m['data']['end_idx']}",
                "preview": m['data']['text'][:120] + "…",
                "score": round(float(score), 3),
            })

    return topic_results, chunk_results, checkpoint_results, sources


def build_rag_answer(query, topic_results, chunk_results, checkpoint_results):
    """Combine retrieved topic summaries and message chunks into a synthesized answer."""
    if not topic_results and not chunk_results:
        return "No strong evidence was found regarding this topic in the conversations."

    # Build context string
    topic_text = "\n".join(topic_results[:2]) if topic_results else "None"
    chunk_text = "\n".join(chunk_results[:3]) if chunk_results else "None"
    
    context = f"TOPIC SUMMARIES:\n{topic_text}\n\nMESSAGE CHUNKS:\n{chunk_text}"
    context = context[:1500]
    
    prompt = f"""You are a grounded RAG assistant.

Use ONLY the retrieved conversation evidence below.

Rules:
- Do NOT invent facts
- Do NOT speculate
- If evidence is weak, say so
- Keep answers concise and natural

Question:
{query}

Retrieved Evidence:
{context}
"""
    
    # Standard template-based response (Ollama removed for easy cloud deployment)
    parts = [f"Based on the retrieved conversation history, here is what I found regarding **{query}**:"]

    if topic_results:
        parts.append("\n**High-Level Context (Topic Summaries):**")
        parts.append("The users frequently discuss this in the context of:")
        for ts in topic_results[:2]:
            parts.append(f"• {ts}")

    if chunk_results:
        parts.append("\n**Specific Conversation Evidence:**")
        parts.append("Here are the exact excerpts where they mentioned this:")
        for ch in chunk_results[:3]:
            parts.append(f"> \"{ch}\"")

    if not topic_results and not chunk_results:
        return f"I couldn't find any specific information in the conversation history regarding **{query}**."

    return "\n".join(parts)


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/persona")
def get_persona():
    return persona


@app.get("/api/topics")
def get_topics():
    return {"count": len(topics_list), "topics": topics_list[:50]}


@app.get("/api/checkpoints")
def get_checkpoints():
    return {"count": len(checkpoints_list), "checkpoints": checkpoints_list[:20]}


@app.get("/api/health")
def health():
    return {"status": "ok", "index_loaded": index is not None, "topics": len(topics_list), "checkpoints": len(checkpoints_list)}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if index is None or embedder is None:
        return {"response": "System not initialised. Run `python data_pipeline.py && python persona_builder.py` first.", "sources": []}

    query = req.message

    # ── Intent routing ─────────────────────────────────────────────
    q_lower = query.lower()

    # Persona: habits
    if PERSONA_HABITS_KW.search(q_lower):
        answer = format_persona_habits(persona)
        # Also retrieve supporting RAG evidence
        topic_r, chunk_r, cp_r, sources = rag_retrieve(query, k=3)
        if chunk_r:
            answer += "\n\n**Supporting conversation evidence:**\n"
            for i, ch in enumerate(chunk_r[:2], 1):
                answer += f"{i}. {ch}\n"
        return {"response": answer, "sources": sources}

    # Persona: style / "how do they talk"
    if PERSONA_STYLE_KW.search(q_lower):
        answer = format_persona_style(persona)
        return {"response": answer, "sources": [{"type": "persona", "preview": "Communication Style Analysis"}]}

    # Persona: "what kind of person"
    if PERSONA_PERSON_KW.search(q_lower):
        answer = format_persona_full(persona)
        return {"response": answer, "sources": [{"type": "persona", "preview": "Full User Profile"}]}

    # ── General RAG query ──────────────────────────────────────────
    topic_r, chunk_r, cp_r, sources = rag_retrieve(query, k=8)
    answer = build_rag_answer(query, topic_r, chunk_r, cp_r)
    return {"response": answer, "sources": sources}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
