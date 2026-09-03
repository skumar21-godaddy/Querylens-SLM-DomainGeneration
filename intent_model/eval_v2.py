"""eval_v2.py — SPAN-focused validation (spans are the priority; intent secondary).
Merges gold_v2 + gold_v2_extra, auto-filters to UNSEEN, reports per-label P/R/F for the
6 span labels + a macro-F1 over the PRIORITY labels, for one or more models. Intent shown
as a secondary line. Applies normalize_case (train==serve).

Usage: python3 eval_v2.py [model_dir1 model_dir2 ...]  (default: model_v2)
"""
from __future__ import annotations
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "intent_layer"))
import spacy
import linguistic_features as LF

HERE = os.path.dirname(os.path.abspath(__file__))
# priority order the user cares about (qualifier included as a span)
LABELS = ["CONCEPT", "GIVEN_NAME", "STYLE", "REQUIRE_TOKEN", "EXCLUDE_TOKEN", "QUALIFIER"]


def norm(q): return re.sub(r"\s+", " ", (q or "").lower()).strip()


def train_set():
    s = set()
    p = os.path.join(HERE, "training_data_v2.jsonl")
    if os.path.exists(p):
        for l in open(p):
            try: s.add(norm(json.loads(l)["text"]))
            except Exception: pass
    return s


def load_gold():
    g = json.load(open(os.path.join(HERE, "gold_v2.json")))
    p = os.path.join(HERE, "gold_v2_extra.json")
    if os.path.exists(p):
        g += json.load(open(p))
    tr = train_set()
    return [x for x in g if norm(x["text"]) not in tr]


def prf(nlp, gold, label):
    tp = fp = fn = 0
    for g in gold:
        gs = {s["text"].lower().strip() for s in g.get("spans", []) if s["label"] == label}
        ps = {e.text.lower().strip() for e in nlp(LF.normalize_case(g["text"])).ents if e.label_ == label}
        tp += len(ps & gs); fp += len(ps - gs); fn += len(gs - ps)
    P = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    R = tp / (tp + fn) if tp + fn else 1.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp + fn


def main():
    models = sys.argv[1:] or ["model_v2/model-best"]
    gold = load_gold()
    print(f"span gold (unseen): {len(gold)}\n")
    for md in models:
        path = md if os.path.exists(md) else os.path.join(HERE, md)
        if not os.path.exists(path):
            print(f"{md}: (missing)\n"); continue
        nlp = spacy.load(path)
        print(f"===== {md} =====")
        print(f"{'label':14s} {'P':>5s} {'R':>5s} {'F1':>5s} {'#':>4s}")
        fs = []
        for lab in LABELS:
            P, R, F, n = prf(nlp, gold, lab)
            fs.append(F)
            print(f"{lab:14s} {P:5.2f} {R:5.2f} {F:5.2f} {n:4d}")
        macro = sum(fs) / len(fs)
        iok = sum(1 for g in gold if max(nlp(LF.normalize_case(g["text"])).cats,
                                         key=nlp(LF.normalize_case(g["text"])).cats.get) == g["intent"])
        print(f"{'MACRO span F1':14s} {macro:5.2f}   (intent, secondary: {iok/len(gold):.2f})\n")


if __name__ == "__main__":
    main()
