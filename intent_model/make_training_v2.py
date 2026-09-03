"""make_training_v2.py — build the spaCy DocBin from the single consolidated dataset
`training_data_v2.jsonl` (6-label schema). Self-contained: no external label deps.

- normalize_case at TRAIN time (case + whitespace) == serve
- SPAN_MODE (default): keep ALL data + oversample rare labels (EXCLUDE x3, REQUIRE x2)
  → best span scores (spans are the priority). Intent is left as-is (secondary).
- writes train_v2.spacy / dev_v2.spacy (8% dev), prints composition.
"""
from __future__ import annotations

import json, os, random, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "intent_layer"))
import linguistic_features as LF
random.seed(17)

DATA = os.path.join(HERE, "training_data_v2.jsonl")
INTENTS = ["creative", "exact", "ambiguous"]
SPAN_LABELS = {"CONCEPT", "STYLE", "GIVEN_NAME", "QUALIFIER", "REQUIRE_TOKEN", "EXCLUDE_TOKEN"}


def _spans_no_overlap(doc, raw_spans):
    """Map (text,label) → token-aligned Spans; longest-first; drop overlaps
    (NER needs disjoint). alignment_mode='contract' avoids grabbing extra tokens."""
    out, used = [], []
    for text, lab in sorted(raw_spans, key=lambda x: -len(x[0])):
        tl = (text or "").lower().strip()
        if not tl:
            continue
        i = doc.text.lower().find(tl)
        if i < 0:
            continue
        j = i + len(tl)
        if any(i < e and s < j for s, e in used):
            continue
        sp = doc.char_span(i, j, label=lab, alignment_mode="contract") \
            or doc.char_span(i, j, label=lab, alignment_mode="expand")
        if sp is not None and not any(sp.start < e2 and s2 < sp.end
                                      for s2, e2 in [(x.start, x.end) for x in out]):
            out.append(sp); used.append((i, j))
    return out


def main():
    data = []
    for l in open(DATA):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("text") and r.get("intent") in INTENTS:
            data.append(r)
    random.shuffle(data)

    def has(r, lab): return any(s.get("label") == lab for s in r.get("spans", []))
    if os.getenv("SPAN_MODE", "1") == "1":
        excl = [r for r in data if has(r, "EXCLUDE_TOKEN")]
        req = [r for r in data if has(r, "REQUIRE_TOKEN")]
        data = data + excl * 2 + req * 1            # EXCLUDE x3, REQUIRE x2
        random.shuffle(data)

    lab_counts = Counter(s["label"] for r in data for s in r.get("spans", [])
                         if s.get("label") in SPAN_LABELS)
    print(f"train rows (after oversample)={len(data)}")
    print(f"intent: {dict(Counter(r['intent'] for r in data))}")
    print(f"span labels: {dict(lab_counts)}")

    import spacy
    from spacy.tokens import DocBin
    nlp = spacy.blank("en")
    n_dev = int(0.08 * len(data))
    dev, train = data[:n_dev], data[n_dev:]

    def build(rows, path):
        db = DocBin()
        for r in rows:
            doc = nlp.make_doc(LF.normalize_case(r["text"]))
            doc.cats = {k: (1.0 if k == r["intent"] else 0.0) for k in INTENTS}
            raw = [(s["text"], s["label"]) for s in r.get("spans", [])
                   if s.get("label") in SPAN_LABELS and s.get("text")]
            doc.ents = _spans_no_overlap(doc, raw)
            db.add(doc)
        db.to_disk(path); return len(rows)

    ntr = build(train, os.path.join(HERE, "train_v2.spacy"))
    ndv = build(dev, os.path.join(HERE, "dev_v2.spacy"))
    print(f"train={ntr} dev={ndv} -> train_v2.spacy / dev_v2.spacy")


if __name__ == "__main__":
    main()
