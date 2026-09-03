# QueryLens

**Query understanding for domain search — intent, spans & constraints from natural language.**

A CPU-only, ~10 ms/query NLP layer that turns a natural-language domain-search query
into a **structured brief** for downstream name generation + ranking. It classifies
**intent** and extracts typed **spans**, trained by distilling a large LLM offline so
serving costs **$0 per request** (no LLM in the request path).

```
"a fun short name for my vegan bakery for kids, avoid the word green, .shop only"
   intent      : creative
   concept     : ["vegan bakery"]
   style       : ["fun", "short"]
   qualifier   : ["kids"]
   exclude_token: ["green"]
   constraints : {"tld": ".shop"}
```

## What it produces
- **intent** (spaCy textcat): `creative` | `exact` | `ambiguous`
- **spans** (spaCy NER): `CONCEPT`, `STYLE`, `GIVEN_NAME`, `QUALIFIER`,
  `REQUIRE_TOKEN`, `EXCLUDE_TOKEN`
- **constraints** (regex, closed-form): `tld`, `length_range/max/min`, `price_max`,
  `no_hyphen`, `no_digits`, `position`

Design principle: the model learns everything semantic/context-dependent; regex handles
only crisp closed-form constraints. **Span quality is the primary objective;** intent is
secondary. Measured quality (held-out gold): **macro span-F1 0.80**, high precision on
every label (see `EVAL_RESULTS_v2.md`).

## Layout (this is the standalone solution)
```
intent_model/
  hybrid.py                 # the analyzer: hybrid.analyze(query) -> dict     [ENTRY POINT]
  model_v2_span/            # the trained spaCy model (model-best = production)
  config_ner.cfg            # spaCy training config (tok2vec CNN + textcat + ner)
  training_data_v2.jsonl    # the single consolidated training dataset (6,935 rows)
  train_v2.spacy / dev_v2.spacy   # built DocBins
  gold_v2.json, gold_v2_extra.json, gold.json   # held-out evaluation gold
  make_training_v2.py       # dataset -> DocBins    | eval_v2.py  # per-label P/R/F
  label_v2.py, gen_synthetic_v2.py   # data-creation reference (need external CSVs/token — see note)
  *.md, EVAL_RESULTS_v2.*   # docs (architecture, operations, guide, backlog, results)
intent_layer/
  constraints.py            # closed-form constraint rules (tld/length/price/…)
  tld_corpus.txt            # ~530 real ASCII TLDs — validates TLD extraction
  linguistic_features.py    # normalize_case (case + whitespace normalization)
demo/
  serve.py, engine.py       # zero-dependency web demo (http://localhost:8000)
```
> `hybrid.py` imports the two `intent_layer` rule files, so keep `intent_model/` and
> `intent_layer/` together. `demo/` is optional (for interactive testing).

## Setup — run the PRETRAINED model (no retraining needed)

The trained model `model_v2_span/model-best` is committed in the repo **with its word
vectors baked in** (`meta.json` requirements = none). So anyone can clone and generate
responses immediately — **serving needs only Python + `spacy`**. No GPU, no LLM/API
token, no `en_core_web_md`/`en_core_web_sm`, and **no Git LFS** (largest file ≈ 33 MB).

```bash
# 1. clone
git clone <repo-url> && cd <repo>/solution      # keep intent_model/ + intent_layer/ + demo/ together

# 2. install (Python 3.9–3.12)
pip install "spacy>=3.8.16,<3.9"                 # that's all serving needs

# 3a. use it in code
python - <<'PY'
import sys; sys.path += ["intent_model", "intent_layer"]
import hybrid
print(hybrid.analyze("suggest a modern name for a coffee shop in austin, avoid ai"))
PY

# 3b. or run the web demo
INTENT_MODEL_DIR=$PWD/intent_model/model_v2_span/model-best python3 demo/serve.py
# open http://localhost:8000
```
First call warms the model (~1–2 s); subsequent calls ~10–13 ms on CPU.

**Extra deps, only if you regenerate labels / retrain from raw** (a normal retrain from
`training_data_v2.jsonl` needs NONE of this): `pip install anthropic`, a CAAS token, and
the original GoDaddy CSVs (not included). Serving and dataset-retraining need only `spacy`.

## Retrain (from the shipped dataset — no external data needed)
```bash
cd intent_model
SPAN_MODE=1 python3 make_training_v2.py
python3 -m spacy train config_ner.cfg --paths.train ./train_v2.spacy \
  --paths.dev ./dev_v2.spacy --output ./model_v2_span --training.max_epochs 40 --system.seed 42
python3 eval_v2.py model_v2_span/model-best
```
See `OPERATIONS.md` for the full runbook and `TRAINING_DATA_GUIDE.md` for how to create
training data (schema, balancing, synthetic augmentation, pitfalls).

## Documentation
| file | purpose |
|---|---|
| `MODEL_ARCHITECTURE_AND_TRAINING.md` | architecture, math, training process (for ML scientists) |
| `OPERATIONS.md` | file execution flow, model storage, retrain & hosting runbook |
| `TRAINING_DATA_GUIDE.md` | how to create training data (the playbook) |
| `TRAINING_DATA_BACKLOG.md` | remaining patterns to improve (R1–R6) |
| `NEXT_STEPS.md` | patterns A–I + model-optimization experiments |
| `TRAINING_DATA_V2_PLAN.md` | the v2 data-build plan (as executed) |
| `EVAL_RESULTS_v2.md` / `.json` | evaluation results (per-label + per-query) |

## Bring your own LLM key (only for data creation / synthetic / validation data)
Serving and dataset-retraining need **no** API key. The LLM teacher is used **only** to
create new labels — synthetic queries (`gen_synthetic_v2.py`), fresh training data, or
validation data (`label_v2.py`). No key is committed to this repo; set your own:
```bash
export CAAS_AUTH_TOKEN=<your-own-llm-api-key>
export CAAS_BASE_URL=<your-llm-endpoint>     # e.g. an Anthropic-compatible endpoint
export CAAS_MODEL=<model-id>                 # optional; defaults to claude-opus-4-8
```
`label_v2.py` reads all three from the environment — no key or endpoint is hardcoded.

## Notes / limitations
- **Data creation vs retraining:** `training_data_v2.jsonl` is the labeled artifact;
  retraining needs only it. Regenerating labels from raw (`label_v2.py` /
  `gen_synthetic_v2.py`) requires the original GoDaddy query CSVs (**not included**) and
  a CAAS `CAAS_AUTH_TOKEN`.
- **Known weak spot:** `EXCLUDE_TOKEN` recall (~0.42) — negation is the hardest label
  (precision stays ~0.91). See `TRAINING_DATA_BACKLOG.md` R5.
- Trained/evaluated on **non-purchase** data; purchase impact is an online A/B, not claimed here.
- Serving is CPU-only, ~120 MB/process, no GPU, no per-request LLM.
