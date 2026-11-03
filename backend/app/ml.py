"""Severity triage inference — TF-IDF + XGBoost, trained by train_model.py.

This is the sandboxed stand-in for the DistilBERT-based triage component
described in the paper (Section III-A, item 2): this environment has no
internet access to huggingface.co, so a TF-IDF + gradient-boosted-tree
classifier is used instead. It is drop-in replaceable — see README.
"""
import json
import os

import joblib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(HERE, "models")

_model = None
_vectorizer = None
_label_encoder = None
_metrics = None

# severity label -> numeric score used inside the dispatch formula's Sev(i) term
SEVERITY_SCORE = {"critical": 1.0, "moderate": 0.55, "low": 0.2}


def _load():
    global _model, _vectorizer, _label_encoder, _metrics
    if _model is None:
        _model = joblib.load(os.path.join(MODELS_DIR, "triage_model.joblib"))
        _vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        _label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
        with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
            _metrics = json.load(f)
    return _model, _vectorizer, _label_encoder


def get_metrics():
    _load()
    return _metrics


def classify_severity(transcript: str):
    """Returns (label, confidence, sev_score) for a raw incident transcript."""
    model, vectorizer, le = _load()
    X = vectorizer.transform([transcript])
    proba = model.predict_proba(X)[0]
    idx = proba.argmax()
    label = le.inverse_transform([idx])[0]
    confidence = float(proba[idx])
    sev_score = SEVERITY_SCORE.get(label, 0.5) * (0.6 + 0.4 * confidence)
    return label, confidence, round(sev_score, 4)
