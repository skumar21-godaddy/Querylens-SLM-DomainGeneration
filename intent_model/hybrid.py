"""hybrid.py — production intent + span analyzer (v2).

analyze(query) -> dict. Combines:
  intent, concept, style, given_name, qualifier, require_token, exclude_token
      → the trained spaCy model (textcat + NER), verbatim.
  constraints (tld / length / price / no-hyphen / no-digit / position)
      → deterministic regex rules in intent_layer/constraints.py.

Preprocessing: intent_layer/linguistic_features.normalize_case (all-caps→lower,
collapse whitespace), applied identically at train and serve time.

CPU-only, ~10-13 ms/query, no per-request LLM. The model directory is self-contained
(word vectors bundled); override the default with env INTENT_MODEL_DIR.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "intent_layer"))

import constraints as CN                     # closed-form constraint rules (regex)
from linguistic_features import normalize_case

_model = None
_MODEL_DIR = os.environ.get("INTENT_MODEL_DIR") or os.path.join(HERE, "model_v2_span", "model-best")


def _load():
    global _model
    if _model is None:
        import spacy
        _model = spacy.load(_MODEL_DIR)
    return _model


def analyze(query: str) -> dict:
    """Return the structured brief for a domain-search query."""
    doc = _load()(normalize_case(query or ""))
    intent = max(doc.cats, key=doc.cats.get) if doc.cats else "exact"

    ents = {}
    for e in doc.ents:
        ents.setdefault(e.label_, []).append(e.text)

    # closed-form constraints from regex; the model owns require/exclude tokens.
    cons, _ = CN.extract(query)
    cons.pop("required_token", None)
    cons.pop("exclude_token", None)
    require = list(ents.get("REQUIRE_TOKEN", []))
    exclude = list(ents.get("EXCLUDE_TOKEN", []))
    if require:
        cons["require_token"] = require
    if exclude:
        cons["exclude_token"] = exclude

    return {
        "query": query,
        "intent": intent,                         # MODEL (textcat)
        "concept": list(ents.get("CONCEPT", [])),        # MODEL (NER)
        "style": list(ents.get("STYLE", [])),            # MODEL (NER)
        "given_name": list(ents.get("GIVEN_NAME", [])),  # MODEL (NER)
        "qualifiers": list(ents.get("QUALIFIER", [])),   # MODEL (NER)
        "require_token": require,                        # MODEL (NER)
        "exclude_token": exclude,                        # MODEL (NER)
        "constraints": cons,                             # REGEX (closed-form)
        "model_ents": ents,                              # every raw label, for inspection
    }


if __name__ == "__main__":
    import json, sys
    queries = sys.argv[1:] or [
        "suggest a fun short name for my vegan bakery for kids",
        "a domain for data engineering but avoid the word ai",
        "The Lounge Coffee Co",
        "flower shop in canada, max 12 characters, .shop only"]
    for q in queries:
        print(json.dumps(analyze(q), ensure_ascii=False, indent=2))
