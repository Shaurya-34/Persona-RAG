# System Design — Local-First Sync Architecture

## Overview

The Persona-RAG system follows a **local-first** architecture: all inference, retrieval, and persona tracking run entirely on-device. Sync is optional and additive — the system never depends on connectivity for core functionality.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT DEVICE                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  FAISS Index  │  │ Intent Model │  │  Persona Engine  │  │
│  │  (88 MB)      │  │  (75 KB)     │  │  (timeline.json) │  │
│  │  rag_index    │  │  TF-IDF +    │  │  per-day drift   │  │
│  │  .faiss       │  │  LogReg      │  │  detection       │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │             │
│         ▼                 ▼                    ▼             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              FastAPI Application Layer               │    │
│  │  • Conflict-aware retrieval (recency + emotion)      │    │
│  │  • Intent classification (<0.11ms per message)        │    │
│  │  • Contradiction detection across chunks              │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                    │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │              Local Storage (data_cache/)              │    │
│  │  messages.json │ topics.json │ timeline.json          │    │
│  │  persona.json  │ day_stats.json │ checkpoints.json    │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                    │
│                    SYNC BOUNDARY                             │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│                         │                                    │
│                   (optional)                                 │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    REMOTE (Optional)  │
              │  ┌─────────────────┐  │
              │  │ Object Storage  │  │
              │  │ (S3 / GCS)     │  │
              │  │ • Encrypted     │  │
              │  │   persona       │  │
              │  │   snapshots     │  │
              │  │ • Timeline      │  │
              │  │   events        │  │
              │  └─────────────────┘  │
              │  ┌─────────────────┐  │
              │  │ Metadata DB     │  │
              │  │ (Postgres/      │  │
              │  │  SQLite Cloud)  │  │
              │  │ • Sync cursors  │  │
              │  │ • Device registry│  │
              │  │ • Conflict log  │  │
              │  └─────────────────┘  │
              └───────────────────────┘
```

---

## What Stays Local vs. What Syncs

| Data | Stays Local | Syncs | Rationale |
|---|---|---|---|
| Raw messages (`messages.json`, 25 MB) | ✅ | ❌ | Too large, privacy-sensitive, only needed for re-indexing |
| FAISS index (`rag_index.faiss`, 88 MB) | ✅ | ❌ | Device-specific binary, rebuilt from messages |
| Intent model (`intent_model.pkl`, 75 KB) | ✅ | ❌ | Deterministic, same model on all devices |
| Persona snapshot (`persona.json`, 5 KB) | ✅ | ✅ | Small, useful for cross-device continuity |
| Timeline events (`timeline.json`) | ✅ | ✅ | Critical for drift history, small payload |
| Day stats (`day_stats.json`) | ✅ | ❌ | Derived data, can be recomputed |
| New conversation chunks | ✅ | ✅ | Incremental — only new chunks sync |
| Topic summaries | ✅ | ❌ | Derived from messages, recomputable |

**Design Principle**: Sync the *conclusions* (persona, timeline, drift events), not the *raw data* (messages, embeddings).

---

## Conflict Resolution Strategy

### Problem
Two devices may process new conversations independently, producing divergent persona snapshots or conflicting timeline entries.

### Solution: Last-Write-Wins with Semantic Merge

```
Device A: persona.sentiment = 0.6  (updated at T=10)
Device B: persona.sentiment = 0.3  (updated at T=12)

Resolution: Device B wins (T=12 > T=10)
```

**For simple scalar fields** (sentiment score, tone label, message counts):
- **Last-Write-Wins (LWW)** using wall-clock timestamps per field
- Each sync payload carries a `updated_at` timestamp per key

**For append-only data** (timeline events, drift entries):
- **Union merge** — all events from both devices are kept
- Deduplicated by `(day_id, drift_type)` composite key
- No data is ever deleted during sync

**For FAISS index**:
- Never synced — each device rebuilds locally from its own messages
- If a device receives new synced conversation chunks, it triggers a local re-index

### Why Not Vector Clocks?

Vector clocks add complexity appropriate for collaborative editing (Google Docs), but persona data is:
1. **Append-mostly** — new days are added, old days rarely change
2. **Convergent** — both devices processing the same messages will produce the same persona
3. **Low-conflict** — only one user typically adds conversations at a time

LWW + union merge handles 99% of real-world cases with zero overhead.

---

## On-Device Storage Options

| Platform | Storage Engine | Capacity | Notes |
|---|---|---|---|
| **Desktop/Server** | Filesystem (JSON + FAISS) | Unlimited | Current implementation |
| **Mobile (iOS/Android)** | SQLite + FTS5 | ~500 MB practical | Replace FAISS with SQLite FTS for text search |
| **Browser (PWA)** | IndexedDB + OPFS | ~1 GB (varies by browser) | FAISS.js exists but limited; fallback to brute-force cosine |

### Mobile Adaptation
- Swap FAISS for **SQLite FTS5** (full-text search) — no native binary dependency
- Store embeddings as BLOBs in SQLite, do cosine search in-app
- Intent model runs via **ONNX Runtime Mobile** (~2 MB overhead)

---

## Tradeoffs & Design Decisions

| Decision | Pros | Cons |
|---|---|---|
| **Local-first (no cloud dependency)** | Zero latency, full privacy, works offline | No cross-device sync out-of-the-box; user must manually transfer data |
| **TF-IDF + LogReg over transformer** | 75 KB model, 0.11ms inference, zero GPU need | Lower accuracy (78%) vs DistilBERT (~92%); struggles with ambiguous inputs |
| **Synthetic day bucketing (15 convos = 1 day)** | Works without real timestamps; enables temporal features | Not true chronological ordering; drift detection granularity is approximate |
| **LWW conflict resolution over CRDTs** | Simple to implement, no state overhead | Can lose data if two devices edit the same scalar field simultaneously |
| **FAISS over SQLite FTS** | Fast dense vector search, handles semantic similarity | Large binary (60 MB), not portable to mobile/browser without adaptation |
| **Template-based synthesis over LLM** | Fully deterministic, zero API cost, instant response | Answers lack natural fluency; can't rephrase or reason over retrieved context |
| **Composite scoring (cosine + recency + emotion)** | Balances relevance with temporal context and emotional importance | Weight values (0.6/0.25/0.15) are hand-tuned, not learned from user feedback |

---

## Sync Protocol (If Implemented)

```
1. Device computes local_cursor = hash(last_synced_state)
2. POST /sync { cursor, persona_delta, new_timeline_entries }
3. Server merges using LWW + union rules
4. Server returns { merged_persona, merged_timeline, new_cursor }
5. Device applies merge, updates local_cursor
```

- **Transport**: HTTPS REST (simple) or WebSocket (real-time)
- **Auth**: Device-bound JWT tokens
- **Encryption**: All persona data encrypted at rest (AES-256) and in transit (TLS 1.3)
- **Frequency**: On app open + every 15 minutes while active
