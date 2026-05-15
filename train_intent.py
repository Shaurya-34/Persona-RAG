"""
Offline Intent Classifier — Train & Export
============================================
Trains a TF-IDF + Logistic Regression classifier on intent_dataset.csv.
Exports:
  - data_cache/intent_model.pkl      (~1-3 MB)
  - data_cache/intent_vectorizer.pkl (~1-2 MB)

Runs fully offline.  No API calls.
Expected: <50 MB total model size, <200ms inference on CPU.
"""

import csv
import os
import sys
import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


DATASET_PATH = 'intent_dataset.csv'
OUTPUT_DIR = 'data_cache'


def load_dataset(path):
    """Load intent_dataset.csv → (texts, labels)."""
    texts, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'].strip())
            labels.append(row['intent'].strip())
    return texts, labels


def train_and_export():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("  Offline Intent Classifier — Training")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────
    texts, labels = load_dataset(DATASET_PATH)
    unique_labels = sorted(set(labels))
    print(f"\n  Samples:  {len(texts)}")
    print(f"  Classes:  {unique_labels}")
    for lbl in unique_labels:
        count = labels.count(lbl)
        print(f"    {lbl}: {count} examples")

    # ── Vectorize ──────────────────────────────────────────────────────
    print("\n[1/3] TF-IDF Vectorization …")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents='unicode',
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    print(f"  Feature matrix: {X.shape}")

    # ── Train ──────────────────────────────────────────────────────────
    print("\n[2/3] Training Logistic Regression …")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial',
    )
    model.fit(X, y)

    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"  5-fold CV accuracy: {scores.mean():.3f} (±{scores.std():.3f})")

    # ── Benchmark latency ──────────────────────────────────────────────
    test_msg = "Remind me to call the doctor tomorrow"
    test_vec = vectorizer.transform([test_msg])
    start = time.perf_counter()
    for _ in range(1000):
        model.predict(test_vec)
    elapsed = (time.perf_counter() - start) / 1000 * 1000  # ms per inference
    print(f"  Inference latency: {elapsed:.2f}ms per message (target: <200ms)")

    # ── Export ──────────────────────────────────────────────────────────
    print("\n[3/3] Exporting model …")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vec_path = os.path.join(OUTPUT_DIR, 'intent_vectorizer.pkl')
    model_path = os.path.join(OUTPUT_DIR, 'intent_model.pkl')
    joblib.dump(vectorizer, vec_path)
    joblib.dump(model, model_path)

    vec_size = os.path.getsize(vec_path) / 1024
    model_size = os.path.getsize(model_path) / 1024
    print(f"  Vectorizer: {vec_path} ({vec_size:.1f} KB)")
    print(f"  Model:      {model_path} ({model_size:.1f} KB)")
    print(f"  Total:      {(vec_size + model_size):.1f} KB (target: <50 MB)")

    # ── Quick test ─────────────────────────────────────────────────────
    print("\n  Quick sanity check:")
    test_cases = [
        "Remind me to buy groceries",
        "I'm feeling really sad today",
        "We need to deploy the fix by Friday",
        "Hey how's it going",
        "What is the capital of France",
    ]
    for msg in test_cases:
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        conf = max(proba)
        print(f"    \"{msg}\" → {pred} ({conf:.2f})")

    print("\n✓ Intent classifier trained and exported.")


if __name__ == "__main__":
    train_and_export()
