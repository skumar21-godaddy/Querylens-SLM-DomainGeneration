# Demo — Intent Layer (classification only)

A runnable, **offline** demo of the intent layer for team demos. It stops at the
intent decision — **no SLM, no domain generation** — and reports **per-query
latency** and the **compute footprint**.

```
query → intent (exact / creative / ambiguous)
      → concept + style extraction
      → generation_plan (the hand-off contract for downstream SLM)
      + latency (ms) + compute resources
```

The intent layer (`solution/intent_layer/`) is the *real* code.

## Run the web UI (recommended)
```bash
cd find-conversational-search      # repo root
python3 solution/demo/serve.py     # then open http://localhost:8000
```
Pure Python stdlib server — no FastAPI/uvicorn. Type a query (or click an example);
the page shows the **intent** (band, confidence, concept, style, matched cues,
generation plan), the **latency** (median / mean / p95 over many cache-bypassed
calls), and the **compute footprint** (device, cores, peak memory, components).
Set `DEMO_PORT=xxxx` to change the port.

## Or the CLI
```bash
python3 solution/demo/engine.py "suggest a fun name for my vegan bakery"
python3 solution/demo/engine.py "The Lounge Coffee Co"
python3 solution/demo/engine.py "real estate media"
```

## What to show the team
- **"suggest a dominant domain for the gym"** → *creative* (rule override),
  concept=`gym`, style=`dominant`, plan 0/10.
- **"Priority Senior Care Advisors"** → *exact* (named-entity), plan 10/0.
- **"real estate media"** → *ambiguous* → plan **5 exact + 5 creative** (hybrid hand-off).
- **Latency ≈ 3–5 ms/query on CPU** (no GPU); the footprint is dominated by the
  spaCy model, not the classifier (`model.joblib` ≈ 5 MB).

## Measurements (this machine — will vary)
- Latency: **median ~3–5 ms/query**, cache-bypassed (spaCy POS/NER dominates; the
  logistic model is microseconds).
- Compute: **CPU only, no GPU**; peak process memory ~**500 MB** (mostly the loaded
  spaCy `en_core_web_sm` pipeline); model artifact ~5 MB.
- Components: scikit-learn calibrated logistic + spaCy `en_core_web_sm` (POS/NER) + wordfreq.

## Prerequisites (one-time, on the demo machine)
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  scikit-learn scipy spacy wordfreq joblib
python3 -m spacy download en_core_web_sm
```
The trained model (`intent_layer/model.joblib`) + lexicon are committed; if missing,
run `bash solution/intent_layer/run.sh` first.

## Files
`serve.py` (stdlib web server + UI) · `engine.py` (classify + latency + resources).
