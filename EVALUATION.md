# Self-Evaluation Sheet — Round 2

## Candidate Info
- **Project**: Persona-RAG Adaptive Chatbot
- **Round**: 2 (Advanced Features)
- **Date**: 2025-05-15

---

## Part 1: Adaptive Persona Engine

| Requirement | Status | Evidence |
|---|:---:|---|
| Build a persona drift detector | ✅ | `persona_builder.py` — compares per-day sentiment, tone, humor density across 734 simulated days |
| Track mood/tone changes across days | ✅ | `timeline.json` stores per-day features: sentiment (float), tone (str), humor_density, question_ratio |
| Output a timeline (Day 1 → curious, Day 4 → frustrated, etc.) | ✅ | Timeline API (`/api/timeline`) + SVG chart in UI with drift markers |
| Detect trigger for each drift (topic, event, person) | ✅ | TF-IDF keyword extraction + relationship/event signal detection for each drift event |

**Drift events detected**: 2
- Day 289: casual → playful (Δsentiment = 0.172) — triggers: `fun`, `great`, `like`
- Day 733: enthusiastic → casual (Δsentiment = 0.163) — triggers: `doing`, `great`, `like`

---

## Part 2: Offline Intent Classifier

| Requirement | Status | Evidence |
|---|:---:|---|
| Train/fine-tune a lightweight model (<50 MB) | ✅ | TF-IDF + Logistic Regression = **75.4 KB** total (0.15% of budget) |
| Runs fully offline | ✅ | `joblib` serialized sklearn model — zero network calls |
| Classify into: reminder / emotional-support / action-item / small-talk / unknown | ✅ | 5-class classifier with 151 training samples, 78.1% CV accuracy |
| No OpenAI/Gemini API calls | ✅ | Pure scikit-learn — no imports from openai, google, anthropic |
| Runs on CPU in under 200ms per message | ✅ | Benchmarked at **0.11ms** per inference (1,800× under budget) |

**Sanity check results**:
- "Remind me to buy groceries" → `reminder` (59%)
- "I'm feeling really sad today" → `emotional-support` (46%)
- "We need to deploy the fix by Friday" → `action-item` (49%)
- "Hey how's it going" → `small-talk` (44%)
- "What is the capital of France" → `unknown` (54%)

---

## Part 3: Conflict Resolution in RAG

| Requirement | Status | Evidence |
|---|:---:|---|
| Handle "sister" appearing across multiple checkpoints | ✅ | Query retrieves chunks from Days 665, 624, 599, 550, 537, 510 — all mentioning siblings |
| Rank chunks by recency + emotional weight | ✅ | Composite score: `0.6 × cosine + 0.25 × recency + 0.15 × emotional_weight` |
| Flag contradictions | ✅ | `detect_contradictions()` groups by entity, flags when positive + negative sentiment coexist |
| Return a merged coherent answer | ✅ | `build_rag_answer()` merges topic summaries + chunks (recency-ordered) + contradiction warnings |

---

## Part 4: System Design (Local-First Sync)

| Requirement | Status | Evidence |
|---|:---:|---|
| On-device storage design | ✅ | `SYSTEM_DESIGN.md` — filesystem (JSON + FAISS) for desktop, SQLite FTS5 for mobile, IndexedDB for PWA |
| Sync logic (what syncs vs. stays local) | ✅ | Table in doc: raw messages stay local; persona/timeline sync; FAISS rebuilt locally |
| Conflict resolution strategy | ✅ | Last-Write-Wins for scalars, Union Merge for append-only data (timeline events) |
| Architecture diagram | ✅ | ASCII diagram showing client device → sync boundary → optional remote storage |
| Written 1-page doc | ✅ | `SYSTEM_DESIGN.md` — comprehensive single document |

---

## Deliverables Checklist

| Deliverable | Status | Location |
|---|:---:|---|
| GitHub Repo (public, clean) | ✅ | `Shaurya-34/Persona-RAG` |
| Working demo link | ✅ | HuggingFace Spaces / localhost:7860 |
| System Design Doc (1 page) | ✅ | `SYSTEM_DESIGN.md` |
| README with architecture explanation | ✅ | `README.md` |
| Loom walkthrough (explain decisions) | ⬜ | Pending recording |

---

## Technical Metrics Summary

| Metric | Value |
|---|---|
| Total messages processed | 191,853 |
| FAISS index documents | 41,321 |
| Topics detected | 1,031 |
| Checkpoints | 1,919 |
| Simulated days | 734 |
| Drift events | 2 |
| Intent model size | 75.4 KB |
| Intent inference latency | 0.11 ms |
| Intent CV accuracy | 78.1% |
| External API calls (inference) | 0 |

---

## What I Would Improve With More Time

1. **Richer training data** — 151 examples is minimal; 1,000+ would push accuracy above 90%
2. **Transformer-based intent model** — DistilBERT (~65 MB) would handle edge cases better while staying under 50 MB budget
3. **Real timestamps** — synthetic day bucketing limits drift detection granularity
4. **LLM synthesis** — current template-based answers are functional but lack natural fluency; a small local LLM (Phi-3-mini) could improve answer quality
5. **Actual sync implementation** — the design doc is complete but the sync protocol is not implemented yet
