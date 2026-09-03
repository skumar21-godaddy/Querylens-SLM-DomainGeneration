# QueryLens → SLM Domain Generation

End-to-end domain-name generation: **QueryLens** turns a natural-language query into a
structured brief (intent + concept/style/given-name/qualifier/require-exclude +
constraints), and a small **SLM (Qwen2.5-3B)** turns that brief into 10 tailored,
constraint-compliant domain names. A deterministic layer enforces the constraints the
small model is unreliable on.

> Sibling of the standalone **QueryLens** repo. This one adds the SLM generation step
> (`slm_generate.py`) + a generation-aware demo. The core QueryLens model is identical.

```
query ─▶ QueryLens brief ─▶ SLM (system prompt + brief) ─▶ 10 domains ─▶ enforce(constraints)
```

## Layout
- `intent_model/` — QueryLens model (`model_v2_span/`), analyzer (`hybrid.py`), docs.
- `intent_layer/` — `constraints.py`, `tld_corpus.txt`, `linguistic_features.py`.
- `slm_generate.py` — QueryLens brief → SLM → enforced 10 domains (CLI + importable).
- `demo/` — web demo showing the brief, the generated domains, and total generation time.

## Setup
```bash
pip install "spacy>=3.8.16,<3.9"        # QueryLens (model vectors bundled)
# SLM endpoint (bring your own — no endpoint/token is committed):
export SLM_URL="https://<your-slm-endpoint>/v1/chat/completions"
export SLM_VARIANT="Qwen2-5-3B-Instruct"   # optional
```

## Use
```bash
# CLI — generate 10 domains for a query
python3 slm_generate.py "a fun name for my vegan bakery for kids"
# inspect the exact brief + SLM prompt without calling the SLM:
python3 slm_generate.py --dry "The Lounge Coffee Co"

# web demo (brief + generated domains + total time):
INTENT_MODEL_DIR=$PWD/intent_model/model_v2_span/model-best python3 demo/serve.py  # localhost:8000
```

## How generation is controlled (in `slm_generate.py`)
- **System prompt** (tight, for a 3B): keep the concept word readable, wordplay via whole
  words, style=tone, qualifier=flavor; exact intent / given_name → first 2 are exact
  concatenations; ≥4 of 10 end in `.com`; concept-relevant TLDs; honor require/exclude/
  length/hyphen/digit/position; JSON only.
- **Deterministic `enforce()`**: guarantees the QueryLens constraints regardless of SLM
  slips — valid single TLD (validated against the 531-TLD corpus), `constraints.tld`,
  length, hyphen/digit, require/exclude, dedup, the `.com` floor, and the first-2 exact match.
- Sampling knobs (env): `SLM_TEMP` (default 0.5), `SLM_TOP_P` (default 0.85) — lower =
  more coherent/less garbled, higher = more varied.

## Notes
- Serving QueryLens needs no key. The **SLM step needs your own endpoint** (`SLM_URL`);
  nothing internal is committed.
- The 3B SLM is imperfect on tight constraints (occasional thin names); `enforce()`
  guarantees *validity/compliance*, not elegance. A stronger model or a retry loop polishes it.
- QueryLens docs are in `intent_model/` (architecture, operations, training, eval).
