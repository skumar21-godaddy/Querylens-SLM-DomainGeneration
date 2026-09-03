# QueryLens — Operations Runbook

Practical guide: which file runs when, where the trained model lives, what training
looked like, how to retrain, and how to host. Scope is ONLY this intent+span model
(`solution/intent_model/` + the two `solution/intent_layer/` rule files + `solution/demo/`).

Paths are relative to `solution/intent_model/` unless noted. All Python is CPU-only.
The LLM teacher needs env `CAAS_AUTH_TOKEN` (offline labeling/generation only).

---

## 1. File map & execution order

### A. Build training tensors (from the single consolidated dataset)
| step | file | output |
|---|---|---|
| 1 | `make_training_v2.py` | `training_data_v2.jsonl` → oversample (SPAN_MODE) → `train_v2.spacy`, `dev_v2.spacy` |
|   | env: `SPAN_MODE=1` (default, span priority — keep all data + oversample rare labels) | |

### B. Train
| step | command | output |
|---|---|---|
| 2 | `python3 -m spacy train config_ner.cfg --paths.train ./train_v2.spacy --paths.dev ./dev_v2.spacy --output ./model_v2_span --training.max_epochs 40 --system.seed 42` | `model_v2_span/` |

### C. Evaluate
| step | file | what |
|---|---|---|
| 3 | `eval_v2.py [model_dir]` | per-label P/R/F + macro span-F1 on held-out gold |

### D. Serve
| file | role |
|---|---|
| `hybrid.py` | production analyzer: model spans + regex constraints → structured dict |
| `../intent_layer/constraints.py` | closed-form constraints (tld/length/price/…) |
| `../intent_layer/tld_corpus.txt` | ~530 real ASCII TLDs — validates TLD extraction |
| `../intent_layer/linguistic_features.py` | `normalize_case` preprocessing |
| `../demo/serve.py` + `engine.py` | zero-dep web demo (localhost:8000) |

### Data-creation reference (needs EXTERNAL GoDaddy CSVs + CAAS token — not runnable standalone)
| file | role |
|---|---|
| `label_v2.py` | the LLM teacher: SYSTEM prompt (6-label schema contract) + `label_batch` |
| `gen_synthetic_v2.py` | LLM synthetic-query generation across pattern buckets |
The labeled artifact `training_data_v2.jsonl` is provided, so a normal retrain (step 1→)
needs neither these scripts nor the raw CSVs.

**Dependency edges (import graph):** `hybrid → constraints, linguistic_features.normalize_case`;
`make_training_v2 → linguistic_features.normalize_case` (span-align helper inlined);
`eval_v2 → linguistic_features.normalize_case`; `gen_synthetic_v2 → label_v2`.
No cross-file v1 dependencies remain; serving needs only `spacy` + the bundled model.

---

## 2. Where the trained model is stored

- **Production model:** `solution/intent_model/model_v2_span/` (~77 MB).
  - `model-best/` — best dev checkpoint (**use this**; self-contained, vectors bundled).
- `hybrid.py` default points here:
  `_MODEL_DIR = env INTENT_MODEL_DIR or <HERE>/model_v2_span/model-best`.
- Override at runtime with env `INTENT_MODEL_DIR=/path/to/model-best` (used for A/B
  and for serving a stable snapshot while another model trains).
- A `model-best` dir is a self-contained spaCy pipeline (config.cfg, tok2vec/textcat/ner
  weights, vocab, vectors). Ship the whole directory.

---

## 3. What training looked like (v2 span model)

- Data: **7,822** examples (6,135 real relabeled + 800 synthetic, EXCLUDE ×3/REQUIRE ×2
  oversample), intent creative-heavy (span priority), median length 10.
- Config: `config_ner.cfg` (tok2vec CNN 256×8 + textcat ensemble + ner BILOU), Adam
  lr 0.001, dropout 0.1, patience 1600, max_epochs 40, seed 42.
- Run: ~40 epochs, early-plateau; **best dev score 0.77** (combined 0.5·catsF+0.5·entsF)
  at ~step 4,200–6,000; dev intent ~80, dev NER-F ~73. ~single-CPU, tens of minutes.
- Held-out eval: **macro span-F1 0.80** (CONCEPT .83, GIVEN_NAME .96, STYLE .85,
  REQUIRE .81, QUALIFIER .79, EXCLUDE .57; intent 0.92 on span gold, ~0.68 on the
  intent-only neutral hand-gold — intent is the accepted-secondary axis).

---

## 4. How to retrain (reproduce or improve)

**Standard retrain (from the shipped labeled dataset — no external data needed):**
```bash
cd solution/intent_model
SPAN_MODE=1 python3 make_training_v2.py          # training_data_v2.jsonl -> train_v2/dev_v2.spacy
python3 -m spacy train config_ner.cfg \
  --paths.train ./train_v2.spacy --paths.dev ./dev_v2.spacy \
  --output ./model_v2_span --training.max_epochs 40 --system.seed 42
python3 eval_v2.py model_v2_span/model-best      # macro span-F1 on held-out gold
```

**Regenerating the dataset from raw (needs EXTERNAL data — NOT in this repo):** the
original GoDaddy query CSVs (`nonpurchase_*.csv`, `6plus_word_queries.csv`) + a CAAS
`CAAS_AUTH_TOKEN`. The teacher `label_v2.py` (schema/labeling) + `gen_synthetic_v2.py`
(synthetic queries) produce the labeled JSONL that is consolidated into
`training_data_v2.jsonl`. Not required for a normal retrain.

Rules of thumb (full detail in `TRAINING_DATA_GUIDE.md`): never mix old/new label
schema — relabel everything; keep synthetic ≤12%; `normalize_case` at train == serve;
hold gold out (no leakage); select by macro span-F1; change ONE thing at a time.

Optimization levers (measure one at a time — `NEXT_STEPS.md` §3d): `en_core_web_lg`/
floret vectors (OOV/typos), NER `hidden_width` 64→128 (lift EXCLUDE), dropout 0.1→0.2.

---

## 5. How to host (this model only)

The layer is a pure-Python function (`hybrid.analyze(query) -> dict`), CPU-only, no GPU,
no per-request LLM. Options:

**a) Embedded (simplest):** import `hybrid` in your service, call `analyze()`. Ensure
`solution/intent_model` and `solution/intent_layer` are on `sys.path`; the process needs
only `spacy` installed and the `model_v2_span/model-best` directory present (its word
vectors are bundled — no `en_core_web_md` needed). First call warms the model (~1–2 s);
subsequent calls ~10–13 ms.

**b) Reference HTTP service (as shipped):**
```bash
cd solution/demo
INTENT_MODEL_DIR=/abs/path/to/model_v2_span/model-best DEMO_PORT=8000 python3 serve.py
# POST /api/analyze {"query": "..."} -> JSON {intent, concept, style, given_name,
#   qualifiers, require_token, exclude_token, constraints, model_ents, latency, resources}
```
This is stdlib `http.server` (demo-grade). For production, wrap `hybrid.analyze` in
FastAPI/uvicorn (or the existing `router/v1.py::recommend_slm` path) behind a flag.

**Deployment checklist:**
- Package the `model-best/` directory (self-contained) with the service image.
- Pin `INTENT_MODEL_DIR` to that path.
- Dependencies: `spacy` only (vectors bundled in the model); the two `intent_layer`
  files; `hybrid.py`. No `en_core_web_md`, no `anthropic`/`CAAS_AUTH_TOKEN` at serve time.
- Warm the model at startup (load once, reuse — it's cached in-process).
- CPU sizing: ~120 MB RSS/process, ~10–13 ms/query; scale horizontally by workers.
- LRU-cache by normalized query if traffic repeats.

---

## 6. Artifacts inventory (current standalone contents)
- **Model/serve:** `model_v2_span/`, `config_ner.cfg`, `hybrid.py`, `intent_layer/`
  {`constraints.py`, `tld_corpus.txt`, `linguistic_features.py`}, `demo/`.
- **Dataset/tensors:** `training_data_v2.jsonl` (single consolidated dataset),
  `train_v2.spacy`, `dev_v2.spacy`.
- **Gold:** `gold_v2.json`, `gold_v2_extra.json`, `gold.json` (neutral intent ref).
- **Pipeline:** `make_training_v2.py`, `eval_v2.py`; data-creation reference
  `label_v2.py`, `gen_synthetic_v2.py` (need external CSVs/token — not runnable standalone).
- **Docs:** `README.md`, `MODEL_ARCHITECTURE_AND_TRAINING.md`, `OPERATIONS.md`,
  `TRAINING_DATA_GUIDE.md`, `TRAINING_DATA_BACKLOG.md`, `NEXT_STEPS.md`,
  `TRAINING_DATA_V2_PLAN.md`, `EVAL_RESULTS_v2.{md,json}`.
