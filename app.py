"""
FastAPI Chatbot — Conflict-Aware RAG + Offline Intent + Adaptive Persona
==========================================================================
Combines:
  1. Offline intent classification (TF-IDF + LogReg, replaces regex)
  2. Conflict-aware retrieval (recency + emotional weight + contradiction flagging)
  3. Per-day persona timeline & drift events
  4. Global persona habits/style/facts (backward-compatible)
"""

import json
import faiss
import pickle
import os
import re
import numpy as np
import joblib
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

    # New: timeline & day_stats
    timeline = []
    day_stats = []
    if os.path.exists('data_cache/timeline.json'):
        with open('data_cache/timeline.json', 'r', encoding='utf-8') as f:
            timeline = json.load(f)
    if os.path.exists('data_cache/day_stats.json'):
        with open('data_cache/day_stats.json', 'r', encoding='utf-8') as f:
            day_stats = json.load(f)

    print("Loaded successfully.")
except Exception as e:
    print(f"Failed to load data: {e}.  Run data_pipeline.py + persona_builder.py first.")
    index = None
    metadata = []
    persona = {}
    topics_list = []
    checkpoints_list = []
    timeline = []
    day_stats = []
    embedder = None

# ── Load Offline Intent Classifier ────────────────────────────────────
intent_model = None
intent_vectorizer = None
try:
    intent_vectorizer = joblib.load('data_cache/intent_vectorizer.pkl')
    intent_model = joblib.load('data_cache/intent_model.pkl')
    print("Intent classifier loaded.")
except Exception as e:
    print(f"Intent classifier not found ({e}). Falling back to regex.")


class ChatRequest(BaseModel):
    message: str


# ── Intent Classification ──────────────────────────────────────────────
# Regex fallbacks (used only if ML model not loaded)
PERSONA_HABITS_KW   = re.compile(r'\b(habit|sleep|food|eat|exercise|hobby|hobbi|routine|diet)\b', re.I)
PERSONA_STYLE_KW    = re.compile(r'\b(talk|speak|communication|style|write|tone|emoji|message)\b', re.I)
PERSONA_PERSON_KW   = re.compile(r'\b(person|personality|kind of|type of|who is|what is|trait|character)\b', re.I)


def classify_intent(text):
    """Classify user message intent using offline ML model, fallback to regex."""
    if intent_model is not None and intent_vectorizer is not None:
        vec = intent_vectorizer.transform([text])
        prediction = intent_model.predict(vec)[0]
        proba = intent_model.predict_proba(vec)[0]
        confidence = float(max(proba))
        return prediction, confidence

    # Regex fallback
    q_lower = text.lower()
    if re.search(r'\b(remind|remember|forget|alarm|alert|schedule)\b', q_lower):
        return "reminder", 0.7
    if re.search(r'\b(sad|stressed|anxious|worried|lonely|overwhelm|depress|hurt|cry)\b', q_lower):
        return "emotional-support", 0.7
    if re.search(r'\b(need to|have to|must|should|deploy|fix|send|submit|complete|finish)\b', q_lower):
        return "action-item", 0.7
    if re.search(r'\b(hey|hi|how are|what\'s up|good morning|nice|cool|lol|haha)\b', q_lower):
        return "small-talk", 0.7
    return "unknown", 0.5


# ── Persona Formatters ─────────────────────────────────────────────────

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


# ── Conflict-Aware RAG Retrieval ───────────────────────────────────────

def compute_recency_score(global_idx, max_idx):
    """Normalise global_idx to [0, 1] where 1 = most recent."""
    if max_idx == 0:
        return 0.0
    return global_idx / max_idx


def rag_retrieve(query, k=8):
    """Retrieve relevant documents from FAISS with recency + emotional weighting."""
    q_emb = embedder.encode([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k=k * 2)  # over-fetch for re-ranking

    # Find max global_idx for normalisation
    max_idx = 0
    for m in metadata:
        if m['type'] == 'chunk':
            max_idx = max(max_idx, m['data'].get('end_idx', 0))

    candidates = []
    for score, idx in zip(D[0], I[0]):
        if float(score) < 0.25:
            continue
        if idx >= len(metadata):
            continue
        m = metadata[idx]
        cosine_score = float(score)

        # Compute composite score for chunks
        if m['type'] == 'chunk':
            recency = compute_recency_score(m['data'].get('end_idx', 0), max_idx)
            emo_weight = m['data'].get('emotional_weight', 0.0)
            composite = 0.6 * cosine_score + 0.25 * recency + 0.15 * emo_weight
        else:
            composite = cosine_score

        candidates.append((composite, cosine_score, idx, m))

    # Sort by composite score, take top-k
    candidates.sort(key=lambda x: -x[0])
    candidates = candidates[:k]

    topic_results = []
    chunk_results = []
    checkpoint_results = []
    sources = []

    for composite, cosine_score, idx, m in candidates:
        if m["type"] == "topic":
            topic_results.append(m['data']['summary'])
            sources.append({
                "type": "topic",
                "topic_id": m['data']['topic_id'],
                "msgs": f"{m['data']['start_msg']}–{m['data']['end_msg']}",
                "keywords": m['data'].get('keywords', []),
                "preview": m['data']['summary'][:120] + "…",
                "score": round(composite, 3),
            })
        elif m["type"] == "checkpoint":
            checkpoint_results.append(m['data']['summary'])
            sources.append({
                "type": "checkpoint",
                "checkpoint_id": m['data']['checkpoint_id'],
                "msgs": f"{m['data']['start_msg']}–{m['data']['end_msg']}",
                "preview": m['data']['summary'][:120] + "…",
                "score": round(composite, 3),
            })
        elif m["type"] == "chunk":
            chunk_results.append({
                "text": m['data']['text'],
                "day_id": m['data'].get('day_id', -1),
                "sentiment": m['data'].get('avg_sentiment', 0),
                "emotional_weight": m['data'].get('emotional_weight', 0),
                "start_idx": m['data'].get('start_idx', 0),
                "end_idx": m['data'].get('end_idx', 0),
            })
            sources.append({
                "type": "chunk",
                "day_id": m['data'].get('day_id', -1),
                "msgs": f"{m['data']['start_idx']}–{m['data']['end_idx']}",
                "sentiment": m['data'].get('avg_sentiment', 0),
                "preview": m['data']['text'][:120] + "…",
                "score": round(composite, 3),
            })

    return topic_results, chunk_results, checkpoint_results, sources


# ── Contradiction Detection ────────────────────────────────────────────

ENTITY_RE = re.compile(r'\b(sister|brother|mom|dad|mother|father|wife|husband|girlfriend|boyfriend|friend|boss|colleague)\b', re.I)

def detect_contradictions(chunk_results):
    """
    Group chunks by mentioned entities and detect sentiment contradictions.
    Returns list of contradiction descriptions.
    """
    if len(chunk_results) < 2:
        return []

    # Group by entity
    entity_groups = {}
    for chunk in chunk_results:
        entities = set(ENTITY_RE.findall(chunk['text'].lower()))
        for entity in entities:
            if entity not in entity_groups:
                entity_groups[entity] = []
            entity_groups[entity].append(chunk)

    contradictions = []
    for entity, chunks in entity_groups.items():
        if len(chunks) < 2:
            continue

        # Check for sentiment divergence within same entity
        sentiments = [c['sentiment'] for c in chunks]
        has_positive = any(s > 0.3 for s in sentiments)
        has_negative = any(s < -0.3 for s in sentiments)

        if has_positive and has_negative:
            # Sort by recency (day_id)
            sorted_chunks = sorted(chunks, key=lambda c: c['day_id'])
            earlier = sorted_chunks[0]
            later = sorted_chunks[-1]
            contradictions.append({
                "entity": entity,
                "earlier_day": earlier['day_id'],
                "earlier_sentiment": earlier['sentiment'],
                "earlier_preview": earlier['text'][:80],
                "later_day": later['day_id'],
                "later_sentiment": later['sentiment'],
                "later_preview": later['text'][:80],
            })

    return contradictions


def build_rag_answer(query, topic_results, chunk_results, checkpoint_results):
    """Build answer with conflict awareness."""
    if not topic_results and not chunk_results:
        return "No strong evidence was found regarding this topic in the conversations."

    parts = [f"Based on the retrieved conversation history, here is what I found regarding **{query}**:"]

    if topic_results:
        parts.append("\n**High-Level Context (Topic Summaries):**")
        parts.append("The users frequently discuss this in the context of:")
        for ts in topic_results[:2]:
            parts.append(f"• {ts}")

    if chunk_results:
        # Sort by recency (most recent first)
        sorted_chunks = sorted(chunk_results, key=lambda c: -c.get('day_id', 0))

        parts.append("\n**Specific Conversation Evidence** (ordered by recency):")
        for ch in sorted_chunks[:3]:
            day_label = f"Day {ch['day_id']}" if ch.get('day_id', -1) >= 0 else ""
            parts.append(f"> \"{ch['text']}\" {f'({day_label})' if day_label else ''}")

        # Check for contradictions
        contradictions = detect_contradictions(chunk_results)
        if contradictions:
            parts.append("\n**⚠️ Contradictions Detected:**")
            for c in contradictions:
                parts.append(
                    f"• Regarding **{c['entity']}**: "
                    f"Earlier (Day {c['earlier_day']}, sentiment={c['earlier_sentiment']:+.2f}) — "
                    f"\"{c['earlier_preview']}…\"\n"
                    f"  Later (Day {c['later_day']}, sentiment={c['later_sentiment']:+.2f}) — "
                    f"\"{c['later_preview']}…\""
                )

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


@app.get("/api/timeline")
def get_timeline():
    """Return the per-day persona timeline with drift events."""
    drift_events = [t for t in timeline if t.get('drift')]
    return {
        "total_days": len(timeline),
        "drift_events": len(drift_events),
        "timeline": timeline,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "index_loaded": index is not None,
        "intent_model_loaded": intent_model is not None,
        "topics": len(topics_list),
        "checkpoints": len(checkpoints_list),
        "timeline_days": len(timeline),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    if index is None or embedder is None:
        return {"response": "System not initialised. Run `python data_pipeline.py && python persona_builder.py` first.", "sources": []}

    query = req.message

    # ── Intent classification (ML model or regex fallback) ─────
    intent, confidence = classify_intent(query)
    q_lower = query.lower()

    # ── Persona-specific routing ───────────────────────────────
    # Check for persona queries regardless of intent
    if PERSONA_HABITS_KW.search(q_lower):
        answer = format_persona_habits(persona)
        topic_r, chunk_r, cp_r, sources = rag_retrieve(query, k=3)
        if chunk_r:
            answer += "\n\n**Supporting conversation evidence:**\n"
            for i, ch in enumerate(chunk_r[:2], 1):
                answer += f"{i}. {ch['text']}\n"
        return {"response": answer, "sources": sources, "intent": intent, "confidence": round(confidence, 2)}

    if PERSONA_STYLE_KW.search(q_lower):
        answer = format_persona_style(persona)
        return {"response": answer, "sources": [{"type": "persona", "preview": "Communication Style Analysis"}], "intent": intent, "confidence": round(confidence, 2)}

    if PERSONA_PERSON_KW.search(q_lower):
        answer = format_persona_full(persona)
        return {"response": answer, "sources": [{"type": "persona", "preview": "Full User Profile"}], "intent": intent, "confidence": round(confidence, 2)}

    # ── Intent-aware response routing ──────────────────────────
    if intent == "reminder" and confidence > 0.6:
        # Retrieve any related conversation context
        topic_r, chunk_r, cp_r, sources = rag_retrieve(query, k=4)
        answer = f"📌 **Intent detected: Reminder** (confidence: {confidence:.0%})\n\n"
        if chunk_r:
            answer += "Here's what I found related to this in past conversations:\n"
            for ch in chunk_r[:2]:
                answer += f"> \"{ch['text']}\"\n"
        else:
            answer += "I didn't find related past conversations, but I've noted this as a reminder."
        return {"response": answer, "sources": sources, "intent": intent, "confidence": round(confidence, 2)}

    if intent == "emotional-support" and confidence > 0.6:
        topic_r, chunk_r, cp_r, sources = rag_retrieve(query, k=4)
        answer = f"💙 **I notice you're sharing something emotional.**\n\n"
        if chunk_r:
            answer += "Here are related moments from past conversations:\n"
            for ch in chunk_r[:2]:
                answer += f"> \"{ch['text']}\" (Day {ch.get('day_id', '?')})\n"
        return {"response": answer, "sources": sources, "intent": intent, "confidence": round(confidence, 2)}

    # ── General RAG query (with conflict awareness) ────────────
    topic_r, chunk_r, cp_r, sources = rag_retrieve(query, k=8)
    answer = build_rag_answer(query, topic_r, chunk_r, cp_r)
    return {"response": answer, "sources": sources, "intent": intent, "confidence": round(confidence, 2)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
