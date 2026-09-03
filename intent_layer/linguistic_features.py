"""linguistic_features.py — text preprocessing for the intent+span model.

Only `normalize_case` remains in the serving path: it normalizes degenerate input so
the spaCy model sees clean tokens. Applied identically at train and serve time.
(The former rule-based intent-feature code was removed — intent + spans are now fully
learned by the model; see MODEL_ARCHITECTURE_AND_TRAINING.md.)
"""
from __future__ import annotations

import re


def normalize_case(query: str) -> str:
    """Normalize shouty / degenerate casing and whitespace so spaCy POS/NER and the
    model's case features stay reliable; genuine mixed-case names ("The Lounge Coffee
    Co") are left intact.
      - collapse runs of whitespace to a single space (extra spaces break span
        boundaries, e.g. "Kings  Queens" would truncate a GIVEN_NAME to "Kings"),
      - ALL-CAPS input → lowercase,
      - long fully-title-cased input (>=6 words, every word capitalized) → lowercase
        (that's shouty title-case, not an entity name).
    """
    q = re.sub(r"\s+", " ", (query or "")).strip()
    if not any(c.isalpha() for c in q):
        return q
    if q == q.upper():
        return q.lower()
    words = [w for w in q.split() if any(ch.isalpha() for ch in w)]
    if len(words) >= 6 and all(w[:1].isupper() for w in words):
        return q.lower()
    return q
