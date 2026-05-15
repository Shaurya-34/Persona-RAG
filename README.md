---
title: Persona RAG Chatbot
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# RAG Persona Chatbot — Adaptive Offline-First Architecture

A complete end-to-end **Retrieval-Augmented Generation (RAG)** system with **Adaptive Persona Tracking**, **Offline Intent Classification**, and **Conflict-Aware Retrieval**, built on 11,000+ multi-turn conversations (191k+ messages). Everything runs locally — no external LLM API calls.

---

## 🚀 What's New (Round 2)

| Feature | Description |
|---|---|
| **Adaptive Persona Engine** | Tracks emotional drift across 734 simulated days — detects tone changes with trigger analysis |
| **Offline Intent Classifier** | TF-IDF + Logistic Regression model (75 KB, 0.11ms latency) classifies into 5 intents |
| **Conflict-Aware RAG** | Retrieval ranked by recency + emotional weight; flags contradictions across memory chunks |
| **Timeline Visualizer** | Interactive day-by-day sentiment chart with drift markers and trigger inspection |
| **System Design Doc** | Local-first sync architecture with LWW conflict resolution |

---

## 🧠 Architecture

### 1. Topic Segmentation (Chronological Splitting)

Messages are processed chronologically with **topic drift detection**:

- **Overlapping Sliding Windows**: Rolling windows of 10 messages with stride of 5
- **TF-IDF + Cosine Similarity**: Similarity below `0.15` triggers a new topic boundary
- **Extractive Summarization**: Top 3 TF-IDF-weighted sentences per topic segment
- **Forced Checkpoints**: Every 100 messages as baseline historical anchors

**Result**: 1,031 topics detected across 191,853 messages.

### 2. Temporal Enrichment

Since the raw dataset lacks timestamps, we simulate temporal structure:

- **Synthetic Days**: Every 15 conversations grouped as 1 "day" → **734 simulated days**
- **Per-Message Sentiment**: Deterministic scoring using positive/negative keyword density
- **Emotional Weight**: Intensity measurement based on strong emotional language

### 3. Adaptive Persona Engine

Instead of a single static profile, the system builds **per-day persona features**:

```
Day 1 → sentiment: 0.62, tone: "casual"
Day 289 → sentiment: 0.58, tone: "playful" ← DRIFT (casual → playful)
Day 733 → sentiment: 0.32, tone: "casual" ← DRIFT (enthusiastic → casual)
```

**Drift Detection**: Triggers when:
- Sentiment changes > 30%, OR
- Tone classification changes with sentiment delta > 15%

**Trigger Analysis**: For each drift event, extracts top TF-IDF keywords and relationship/event signals to identify probable causes.

### 4. Offline Intent Classifier

Fully offline, CPU-only intent classification:

| Metric | Value | Requirement |
|---|---|---|
| Model Size | 75.4 KB | < 50 MB ✅ |
| Inference Latency | 0.11 ms | < 200 ms ✅ |
| CV Accuracy | 78.1% | — |
| API Calls | Zero | No OpenAI/Gemini ✅ |

**Classes**: `reminder` · `emotional-support` · `action-item` · `small-talk` · `unknown`

**Architecture**: TF-IDF Vectorizer (unigrams + bigrams) → Logistic Regression

### 5. Conflict-Aware RAG Retrieval

The retrieval engine uses a **composite scoring** formula:

```
final_score = 0.6 × cosine_similarity + 0.25 × recency_score + 0.15 × emotional_weight
```

**Contradiction Detection**:
- Groups retrieved chunks by mentioned entities (sister, brother, mom, etc.)
- Detects sentiment divergence within entity groups
- Outputs merged response: *"You initially mentioned X (Day 3), but more recently (Day 12) you said Y"*

### 6. RAG Synthesis Layer

Template-based deterministic synthesis (no LLM dependency):
- Merges **Topic Summaries** for high-level context
- Injects **Conversation Chunks** ordered by recency
- Flags **Contradictions** when detected
- Every answer traceable to source message indices

---

## 💻 Running Locally

### Prerequisites
- Python 3.10+
- Place `conversations.csv` in the root directory

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build Everything (Data Pipeline → Persona → Intent Model)
```bash
python data_pipeline.py
python persona_builder.py
python train_intent.py
```

This generates `data_cache/` containing:
- FAISS index + metadata
- Enriched messages with `day_id`, `sentiment`, `emotional_weight`
- Per-day timeline with drift events
- Intent classifier model

### 3. Start the Chatbot
```bash
python app.py
```
👉 Open `http://127.0.0.1:7860`

---

## ☁️ Deployment

Ready for **Render**, **Railway**, **Heroku**, or **HuggingFace Spaces** (Docker SDK).

**Build Command:**
```bash
pip install -r requirements.txt && python data_pipeline.py && python persona_builder.py && python train_intent.py
```

**Start Command:**
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

---

## 📁 Project Structure

```
├── app.py                  # FastAPI server — RAG + Intent + Conflict Resolution
├── data_pipeline.py        # Topic segmentation + temporal enrichment + FAISS indexing
├── persona_builder.py      # Global persona + per-day drift detection + timeline
├── train_intent.py         # Offline intent classifier training pipeline
├── intent_dataset.csv      # 151 labeled examples (5 intent classes)
├── SYSTEM_DESIGN.md        # Local-first sync architecture document
├── static/
│   └── index.html          # Chatbot UI with timeline visualizer
├── data_cache/
│   ├── rag_index.faiss     # FAISS vector index
│   ├── metadata.pkl        # Chunk/topic/checkpoint metadata
│   ├── messages.json       # Enriched messages (day_id, sentiment, emotional_weight)
│   ├── topics.json         # Detected topic segments
│   ├── checkpoints.json    # 100-message checkpoints
│   ├── persona.json        # Global user persona profiles
│   ├── timeline.json       # Per-day timeline with drift events
│   ├── day_stats.json      # Aggregated per-day statistics
│   ├── intent_model.pkl    # Trained LogReg classifier
│   └── intent_vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧪 Example Queries

**Query**: *"Did I mention anything about my sister?"*
- Retrieves chunks across **multiple days** (Day 665, 624, 599, 510)
- Orders by **recency** (most recent first)
- Checks for **sentiment contradictions** across mentions
- Shows **day labels** on each excerpt

**Query**: *"What are their habits?"*
- Returns persona profile: sleep, food, exercise, hobbies
- Includes supporting conversation evidence

**Query**: *"Remind me to call the doctor"*
- Intent classified as `reminder` (offline ML model)
- Searches for related past conversation context

---

## 🎥 Demo & Media

### Video Walkthrough
[Watch the Demo Video (Loom)](https://www.loom.com/share/5037df93b96f4ef094e1dab9ba330a9c)

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chatbot UI |
| `/api/chat` | POST | Send message — returns answer + sources + intent |
| `/api/persona` | GET | Global persona profiles |
| `/api/topics` | GET | Detected topic segments |
| `/api/checkpoints` | GET | 100-message checkpoints |
| `/api/timeline` | GET | Per-day timeline with drift events |
| `/api/health` | GET | System status |
