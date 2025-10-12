"""
Train a TF-IDF + LogisticRegression classifier on IMDB via Hugging Face `datasets`.
Saves vectorizer + model to models/ (pickle).
"""

import os
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import numpy as np
import random
import argparse

try:
    from app.task import clean_text
except Exception:
    import re
    from string import punctuation
    URL_RE = re.compile(r'https?://\S+|www\.\S+')
    WHITESPACE_RE = re.compile(r'\s+')
    PUNCT_TO_REMOVE = ''.join(ch for ch in punctuation if ch != "'")
    PUNCT_RE = re.compile('[' + re.escape(PUNCT_TO_REMOVE) + ']')
    def clean_text(s: str) -> str:
        if not s:
            return ""
        s = s.lower()
        s = URL_RE.sub("", s)
        s = PUNCT_RE.sub("", s)
        s = WHITESPACE_RE.sub(" ", s).strip()
        return s
    
def prepare_texts(samples):
    return [clean_text(x) for x in samples]

def main(save_dir="models", sample_frac=1.0, max_samples=None, random_seed=42):
    """
    sample_frac: fraction of train set to use (1.0 = all). Use <1.0 for faster experiments.
    max_samples: cap total number of samples (train+test) if you want a tiny run.
    """
    random.seed(random_seed)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("Loading IMDB dataset")
    ds = load_dataset("imdb")

    # Convert to lists
    train_texts = [t for t in ds['train']['text']]
    train_labels = [int(l) for l in ds['train']['label']]
    test_texts = [t for t in ds['test']['text']]
    test_labels = [int(l) for l in ds['test']['label']]

    if sample_frac < 1.0:
        k = max(1, int(len(train_texts) * sample_frac))
        idx = random.sample(range(len(train_texts)), k)
        train_texts = [train_texts[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]

    if max_samples is not None:
        # limit both train and test combined (take proportionally)
        total = len(train_texts) + len(test_texts)
        if total > max_samples:
            # keep first part of train then test truncated
            keep_train = int((len(train_texts) / total) * max_samples)
            keep_test  = max_samples - keep_train
            train_texts = train_texts[:keep_train]
            train_labels = train_labels[:keep_train]
            test_texts = test_texts[:keep_test]
            test_labels = test_labels[:keep_test]

    # CLean texts
    print("Cleaning texts...")
    X_train_clean = prepare_texts(train_texts)
    X_test_clean = prepare_texts(test_texts)

    # Vectorize (use stop words + ngrams)
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=20000, stop_words='english')
    print("Fitting TF-IDF vectorizer on training data...")
    X_train_vec = vec.fit_transform(X_train_clean)

    # Classifier
    clf = LogisticRegression(solver="saga", max_iter=2000, n_jobs=-1, class_weight="balanced")

    print("Training classifier....")
    clf.fit(X_train_vec, train_labels)

    # Evaluate on test 
    print("Transforming test data and evaluating...")
    X_test_vec = vec.transform(X_test_clean)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(test_labels, y_pred)
    print("Test accuracy: ", acc)
    print("Classification report:")
    print(classification_report(test_labels, y_pred, target_names=["negative", "positive"]))

    # Save
    with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    with open(os.path.join(save_dir, "sentiment_model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    print(f"Saved vectorizer -> {os.path.join(save_dir, 'vectorizer.pkl')}")
    print(f"Saved model     -> {os.path.join(save_dir, 'sentiment_model.pkl')}")

    # Quick sanity checks
    def predict_simple(s):
        t = clean_text(s)
        X = vec.transform([t])
        proba = clf.predict_proba(X)[0]
        lbl = "positive" if proba[1] >= 0.5 else "negative"
        return lbl, float(proba[1])

    samples = [
        "I feel great and productive today",
        "Can't focus and I'm anxious about interviews",
        "That movie was awful and boring",
        "Absolutely loved the performance and ending!"
    ]
    print("\nSanity checks:")
    for s in samples:
        lbl, p = predict_simple(s)
        print(f"  '{s}' -> {lbl} (positive_prob={p:.2f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Use fraction of training set (0.1 for quick runs)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total number of samples (train+test) for a tiny run)")
    parser.add_argument("--save-dir", type=str, default="models")
    args = parser.parse_args()
    main(save_dir=args.save_dir, sample_frac=args.sample_frac, max_samples=args.max_samples)

