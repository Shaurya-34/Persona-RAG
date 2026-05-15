"""
Data Pipeline — Topic Segmentation, Temporal Enrichment, FAISS Index
=====================================================================
Processes all conversations chronologically (message by message).
1. Assigns synthetic day_id (every 15 conversations = 1 day).
2. Computes per-message sentiment and emotional weight.
3. Detects topic changes using TF-IDF cosine drift between sliding windows.
4. Creates 100-message checkpoints (independent of topics).
5. Embeds everything into a FAISS vector index for retrieval.
6. Outputs per-day aggregated statistics for drift detection.
"""

import json
import csv
import io
import os
import sys
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import pickle

# ── Configuration ──────────────────────────────────────────────────────
DATA_PATH = 'conversations.csv'
CHECKPOINT_EVERY = 100
TOPIC_WINDOW = 10        # messages per sliding window
TOPIC_STRIDE = 5         # overlapping slide stride
DRIFT_THRESHOLD = 0.15   # cosine sim below this = topic change
MAX_TOPIC_MSGS = 200     # force a topic break if segment gets too long
CHUNK_SIZE = 5           # messages per fine-grained RAG chunk
CONVS_PER_DAY = 15       # synthetic day grouping

# ── Sentiment & Emotion Patterns ──────────────────────────────────────
POSITIVE_RE = re.compile(
    r'\b(love|happy|excited|amazing|awesome|great|wonderful|fantastic|'
    r'grateful|thankful|glad|enjoy|beautiful|perfect|best|fun|nice|cool)\b', re.I)
NEGATIVE_RE = re.compile(
    r'\b(hate|angry|sad|worried|anxious|scared|afraid|terrible|horrible|'
    r'awful|frustrated|upset|annoyed|miserable|lonely|depressed|stressed)\b', re.I)
EMOTION_INTENSE_RE = re.compile(
    r'\b(hate|love|urgent|important|never|always|worst|best|obsessed|'
    r'desperate|furious|ecstatic|terrified|heartbroken|devastated|thrilled)\b', re.I)


def compute_sentiment(text):
    """Deterministic sentiment: +1 per positive word, -1 per negative, normalised to [-1, 1]."""
    pos = len(POSITIVE_RE.findall(text))
    neg = len(NEGATIVE_RE.findall(text))
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def compute_emotional_weight(text):
    """Density of emotionally intense language (0.0–1.0)."""
    words = text.split()
    if not words:
        return 0.0
    hits = len(EMOTION_INTENSE_RE.findall(text))
    return round(min(hits / max(len(words), 1), 1.0), 3)


def extract_sentences(text):
    """Split text into sentences for extractive summarisation."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]


def summarize_text(text, n=3):
    """TF-IDF-weighted extractive summariser — picks the top-n sentences."""
    sentences = extract_sentences(text)
    if len(sentences) <= n:
        return text.strip()
    try:
        tfidf = TfidfVectorizer(stop_words='english').fit_transform(sentences)
        scores = np.asarray(tfidf.sum(axis=1)).ravel()
        top_idx = scores.argsort()[-n:][::-1]
        top_idx.sort()
        return " ".join(sentences[i] for i in top_idx)
    except Exception:
        return " ".join(sentences[:n])


def extract_keywords(texts, top_n=5):
    """Return the top-n TF-IDF keywords from a list of message texts."""
    blob = " ".join(texts)
    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=top_n)
        tfidf.fit_transform([blob])
        return list(tfidf.get_feature_names_out())
    except Exception:
        # Fallback: most-common non-stop words
        words = re.findall(r'\b[a-z]{4,}\b', blob.lower())
        return [w for w, _ in Counter(words).most_common(top_n)]


# ── Main Pipeline ──────────────────────────────────────────────────────
def run_pipeline():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("  RAG Data Pipeline")
    print("=" * 60)

    # ── 0. Load & flatten messages ─────────────────────────────────────
    print("\n[1/5] Reading CSV …")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw = f.read()

    rows = list(csv.reader(io.StringIO(raw)))
    print(f"  → {len(rows)} conversation rows")

    messages = []
    global_idx = 0
    for conv_idx, row in enumerate(rows):
        if not row:
            continue
        for line in row[0].strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            speaker, text = "Unknown", line
            if ":" in line:
                speaker, text = line.split(":", 1)
                speaker, text = speaker.strip(), text.strip()
            messages.append({
                "global_idx": global_idx,
                "conv_id": conv_idx,
                "day_id": conv_idx // CONVS_PER_DAY,
                "speaker": speaker,
                "text": text,
                "sentiment": compute_sentiment(text),
                "emotional_weight": compute_emotional_weight(text),
            })
            global_idx += 1

    total_days = (messages[-1]['day_id'] + 1) if messages else 0
    print(f"  → {len(messages)} total messages")
    print(f"  → {total_days} simulated days (every {CONVS_PER_DAY} conversations = 1 day)")

    # ── 1. Topic Segmentation ──────────────────────────────────────────
    print("\n[2/5] Topic Segmentation (overlapping sliding-window TF-IDF cosine drift) …")

    topics = []
    if not messages:
        return

    topic_start_idx = messages[0]['global_idx']
    prev_text = " ".join(m['text'] for m in messages[0:TOPIC_WINDOW])

    for i in range(TOPIC_STRIDE, len(messages), TOPIC_STRIDE):
        curr_msgs = messages[i:i + TOPIC_WINDOW]
        if not curr_msgs:
            break
        curr_text = " ".join(m['text'] for m in curr_msgs)

        try:
            vecs = TfidfVectorizer(stop_words='english').fit_transform([prev_text, curr_text])
            sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        except Exception:
            sim = 1.0

        current_topic_length = curr_msgs[-1]['global_idx'] - topic_start_idx

        if sim < DRIFT_THRESHOLD or current_topic_length >= MAX_TOPIC_MSGS:
            # ─── topic boundary detected ───
            # The topic ends right before the new window `i`
            boundary_idx = curr_msgs[0]['global_idx']
            t_msgs = [m for m in messages if topic_start_idx <= m['global_idx'] < boundary_idx]
            if not t_msgs:
                t_msgs = curr_msgs # failsafe
            texts = [m['text'] for m in t_msgs]
            topics.append({
                "topic_id": len(topics),
                "start_msg": t_msgs[0]['global_idx'],
                "end_msg": t_msgs[-1]['global_idx'],
                "message_count": len(t_msgs),
                "keywords": extract_keywords(texts),
                "summary": summarize_text(" ".join(texts), n=3),
            })
            topic_start_idx = boundary_idx

        prev_text = curr_text

    # flush last topic
    if topic_start_idx <= messages[-1]['global_idx']:
        t_msgs = [m for m in messages if m['global_idx'] >= topic_start_idx]
        if t_msgs:
            texts = [m['text'] for m in t_msgs]
            topics.append({
                "topic_id": len(topics),
                "start_msg": t_msgs[0]['global_idx'],
                "end_msg": t_msgs[-1]['global_idx'],
                "message_count": len(t_msgs),
                "keywords": extract_keywords(texts),
                "summary": summarize_text(" ".join(texts), n=3),
            })


    print(f"  → {len(topics)} topics detected")
    for t in topics[:5]:
        print(f"    Topic {t['topic_id']}: msgs {t['start_msg']}–{t['end_msg']} "
              f"({t['message_count']} msgs) keywords={t['keywords']}")
    if len(topics) > 5:
        print(f"    … and {len(topics) - 5} more")

    # ── 2. 100-Message Checkpoints ─────────────────────────────────────
    print("\n[3/5] 100-Message Checkpoints …")
    checkpoints = []
    for i in range(0, len(messages), CHECKPOINT_EVERY):
        chunk = messages[i:i + CHECKPOINT_EVERY]
        chunk_text = " ".join(m['text'] for m in chunk)
        checkpoints.append({
            "checkpoint_id": len(checkpoints),
            "start_msg": chunk[0]['global_idx'],
            "end_msg": chunk[-1]['global_idx'],
            "summary": summarize_text(chunk_text, n=5),
        })
    print(f"  → {len(checkpoints)} checkpoints")

    # ── 3. Embedding & FAISS ───────────────────────────────────────────
    print("\n[4/5] Loading sentence-transformers model …")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []
    metadata = []

    # topic summaries
    for t in topics:
        documents.append(f"Topic about {', '.join(t['keywords'])}. {t['summary']}")
        metadata.append({"type": "topic", "data": t})

    # checkpoint summaries
    for c in checkpoints:
        documents.append(c["summary"])
        metadata.append({"type": "checkpoint", "data": c})

    # fine-grained message chunks (enriched with temporal + sentiment metadata)
    for i in range(0, len(messages), CHUNK_SIZE):
        chunk = messages[i:i + CHUNK_SIZE]
        chunk_text = " ".join(f"{m['speaker']}: {m['text']}" for m in chunk)
        documents.append(chunk_text)
        metadata.append({"type": "chunk", "data": {
            "start_idx": chunk[0]['global_idx'],
            "end_idx": chunk[-1]['global_idx'],
            "day_id": chunk[0]['day_id'],
            "avg_sentiment": round(sum(m['sentiment'] for m in chunk) / len(chunk), 3),
            "emotional_weight": round(sum(m['emotional_weight'] for m in chunk) / len(chunk), 3),
            "text": chunk_text,
        }})

    print(f"\n[5/5] Embedding {len(documents)} documents into FAISS …")
    embeddings = embedder.encode(documents, show_progress_bar=True, batch_size=256)

    index = faiss.IndexFlatIP(embeddings.shape[1])        # inner-product (cosine after normalisation)
    faiss.normalize_L2(embeddings)                        # normalise so IP = cosine
    index.add(embeddings)

    # ── Aggregate per-day statistics (for persona drift engine) ─────────
    print("\n[6/6] Aggregating per-day statistics …")
    from collections import defaultdict
    day_buckets = defaultdict(list)
    for m in messages:
        day_buckets[m['day_id']].append(m)

    day_stats = []
    for day_id in sorted(day_buckets.keys()):
        day_msgs = day_buckets[day_id]
        day_texts = [m['text'] for m in day_msgs]
        n = len(day_msgs)
        question_count = sum(1 for t in day_texts if '?' in t)
        exclaim_count  = sum(1 for t in day_texts if '!' in t)
        avg_sentiment  = round(sum(m['sentiment'] for m in day_msgs) / n, 3)
        avg_emo_weight = round(sum(m['emotional_weight'] for m in day_msgs) / n, 3)
        avg_words      = round(sum(len(t.split()) for t in day_texts) / n, 1)

        day_stats.append({
            "day_id": day_id,
            "message_count": n,
            "avg_sentiment": avg_sentiment,
            "avg_emotional_weight": avg_emo_weight,
            "question_ratio": round(question_count / n, 3),
            "exclamation_ratio": round(exclaim_count / n, 3),
            "avg_words_per_msg": avg_words,
            "keywords": extract_keywords(day_texts, top_n=5),
            "start_idx": day_msgs[0]['global_idx'],
            "end_idx": day_msgs[-1]['global_idx'],
        })
    print(f"  → {len(day_stats)} day records")

    # ── Save ───────────────────────────────────────────────────────────
    os.makedirs('data_cache', exist_ok=True)
    faiss.write_index(index, 'data_cache/rag_index.faiss')
    with open('data_cache/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    with open('data_cache/messages.json', 'w', encoding='utf-8') as f:
        json.dump(messages, f)
    with open('data_cache/topics.json', 'w', encoding='utf-8') as f:
        json.dump(topics, f, indent=2)
    with open('data_cache/checkpoints.json', 'w', encoding='utf-8') as f:
        json.dump(checkpoints, f, indent=2)
    with open('data_cache/day_stats.json', 'w', encoding='utf-8') as f:
        json.dump(day_stats, f, indent=2)

    print("\n✓ Pipeline complete.  data_cache/ populated.")
    print(f"  Topics:      {len(topics)}")
    print(f"  Checkpoints: {len(checkpoints)}")
    print(f"  FAISS docs:  {len(documents)}")
    print(f"  Day stats:   {len(day_stats)}")


if __name__ == "__main__":
    run_pipeline()
