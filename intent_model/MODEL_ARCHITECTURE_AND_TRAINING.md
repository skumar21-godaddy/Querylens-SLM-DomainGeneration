# QueryLens — Architecture, Flow & Training (for ML scientists)

Authoritative technical reference for the GoDaddy domain-search **intent + span
extraction** model (v2, "span-optimized"). Covers the model, the distillation-based
training pipeline, the label schema, evaluation, and every design decision with its
rationale. Serving model: `model_v2_span/` (spaCy pipeline, CPU-only).

---

## 1. Problem & framing

Given a natural-language domain-search query, produce a **structured brief** the
generator + ranker can condition on:
- **intent** (textcat): `creative` | `exact` | `ambiguous`
- **spans** (NER): `CONCEPT`, `STYLE`, `GIVEN_NAME`, `QUALIFIER`, `REQUIRE_TOKEN`, `EXCLUDE_TOKEN`
- **closed-form constraints** (deterministic rules, NOT the model): TLD, length,
  price, no-hyphen/digit, start/end position.

Design priority (explicit): **span quality is the non-negotiable objective**; intent
is secondary (creative/ambiguous confusion tolerated). This drove the training choices
(SPAN_MODE) in §6.

Two jobs, two mechanisms — assigned by *measured* fit, not assumption:
- fuzzy, context-dependent extraction (intent, all spans) → **learned model**
- crisp, closed-form patterns (tld/length/price/…) → **regex** (exact, no data needed)

---

## 2. Serving pipeline (spaCy) — components & tensors

Pipeline: `tok2vec → textcat → ner` (config: `config_ner.cfg`).

### 2.1 Shared encoder `tok2vec` — a CNN (deliberately not a transformer)
```
embed  = MultiHashEmbed(attrs=[NORM, PREFIX, SUFFIX, SHAPE], rows=[5000,1000,2500,2500],
                        include_static_vectors=true, vectors=en_core_web_md 300d)
encode = MaxoutWindowEncoder(width=256, depth=8, window_size=1, maxout_pieces=3)
```
- Each token → concatenation of 4 hashed lexical-attribute embeddings + a 300-d static
  word vector. SUFFIX/SHAPE give partial OOV/typo robustness (`threpaist`≈`therapist`).
- Encoder is a stacked **convolution**: depth 8 × window ±1 ⇒ effective receptive field
  ≈ **±8 tokens** — covers the whole of a typical short query. Maxout (3 pieces) is the
  nonlinearity (max over 3 affine projections per unit).

**Receptive-field note:** because search queries are short (median ≈ 8–10 tokens), the
±8 window already spans most queries, so global (n²) self-attention buys little here —
see §9 for the measured transformer comparison.

### 2.2 Intent head `textcat` = `TextCatEnsemble.v2`
Sum of two sub-models → softmax over 3 exclusive classes:
1. **linear bag-of-words** (`TextCatBOW`, hashed length 262144) — strong lexical cues
   ("suggest", "llc").
2. **CNN** reading the shared tok2vec (`Tok2VecListener`) — word-order/context
   ("company named X" ≠ "name for a company").

`p(kᵢ|q) = softmax(z)ᵢ`, trained with categorical cross-entropy `L=−Σ yₖ log pₖ`.

### 2.3 Span head `ner` = `TransitionBasedParser.v2`
Transition-based **BILOU** tagger (`state_type=ner`, `hidden_width=64`, `maxout=2`),
labels: `CONCEPT, EXCLUDE_TOKEN, GIVEN_NAME, QUALIFIER, REQUIRE_TOKEN, STYLE`. Reads
left→right emitting Begin/In/Last/Unit/Out actions that build **well-formed,
non-overlapping** spans (you can't close a span you didn't open). Chosen over `spancat`
because our spans are short and disjoint — no ngram suggester/threshold needed.

### 2.4 Hyperparameters (config_ner.cfg)
```
optimizer=Adam(β1=.9,β2=.999,L2=.01,grad_clip=1.0)  lr=0.001  dropout=0.1
batch=compounding(100→1000 words)  eval_freq=200  patience=1600  score=0.5·catsF+0.5·entsF
vectors=en_core_web_md   (NOT en_core_web_trf — no transformer in serving)
```

---

## 3. Request flow (hybrid.py)

```
query
  └─ normalize_case()  (all-caps→lower; collapse whitespace runs)   [train==serve]
  └─ spaCy model  → intent (textcat) + spans (ner)
         concept, style, given_name, qualifier, require_token, exclude_token   ← MODEL
  └─ constraints.py (regex, closed-form)  → tld, length_range/max/min, price_max,
         no_hyphen, no_digits, position                                        ← RULES
  └─ merge → { intent, concept[], style[], given_name[], qualifiers[],
               require_token[], exclude_token[], constraints{}, model_ents{} }
```
Serving cost: **~10–13 ms/query, CPU-only, ~120 MB, $0 per request.**

---

## 4. Label schema & disambiguation (the labels the MODEL learns)

| label | definition | tight |
|---|---|---|
| CONCEPT | subject/product/theme/deliverable; every distinct one | 1–3 words |
| STYLE | adjective for the NAME (fun, modern, invented, minimalist) | 1 word |
| GIVEN_NAME | a name the user ALREADY has; the actual proper name only | short; incl. domains, LLC |
| QUALIFIER | audience/location/target-market (kids, canada, barbers) | tight |
| REQUIRE_TOKEN | a word the name MUST contain | 1–2 words |
| EXCLUDE_TOKEN | a word the name must NOT contain (negation, from context) | 1–2 words |

Key learned distinctions: CONCEPT (thing named) vs QUALIFIER (who/where); REQUIRE vs
EXCLUDE (negation in context); GIVEN_NAME = the actual name, never a preamble
("the name shall be **Dignity Moves Foundation**") or a whole aspirational clause.

---

## 5. Training data = LLM distillation (teacher → student)

A strong LLM (Claude `opus`, OFFLINE) labels queries; the cheap spaCy model learns to
imitate and serves. The LLM never runs in production.

**Sources:** `nonpurchase_creative.csv`, `nonpurchase_exact_entity.csv`,
`6plus_word_queries.csv` (long conversational; median 10 words). English-only, deduped.

**Composition (v2 span build ≈ 7,822):**
- ~6,135 real queries relabeled with the 6-label schema (5k diverse existing sample +
  ~1.1k novel ≥10-word 6plus queries)
- ~800 synthetic (LLM-generated, 20–35 words) covering RARE patterns (negation/exclude,
  aspirational, `along with`, comma-less lists) — seeded from real phrasing, typos
  injected, capped ≤12% (drift guard)
- rare labels oversampled (EXCLUDE ×3, REQUIRE ×2) — SPAN_MODE

**Teacher validation:** teacher-vs-hand-gold intent agreement ≈ 0.82 (the realistic
student ceiling). Span labels are the critical artifact — consistency (esp.
CONCEPT-vs-QUALIFIER, REQUIRE-vs-EXCLUDE) determines model quality.

**Consolidated dataset:** all labels live in a single file `training_data_v2.jsonl`
(6,935 unique rows: `{text, intent, spans, source}`, `source∈{real,synthetic}`).

**DocBin build (make_training_v2.py):** read `training_data_v2.jsonl` →
`normalize_case` at train time → SPAN_MODE oversample (EXCLUDE ×3, REQUIRE ×2) →
`doc.cats` one-hot intent, `doc.ents` token-aligned spans (`_spans_no_overlap`:
longest-first, drop overlaps, `alignment_mode="contract"`) → 8% dev split. Gold sets
held out entirely.

---

## 6. Balancing — the key experimental finding

Two objectives conflict; choose by priority:
- **SPAN_MODE (used):** keep ALL data (max span examples), oversample rare labels; do
  NOT intent-downsample. → best span scores.
- **BALANCE:** equal intent (≈33/33/33) recovers intent (~0.80 on neutral gold) but
  downsampling costs span data/recall.

**Measured lessons (documented so they're not repeated):**
- File-proportion re-mixing (e.g. 50/50) without fixing labels → net-negative.
- Intent regressed 0.80→0.68 in early v2 not from encoder sharing (decoupling the
  tok2vec did NOT help) but from **intent label drift** (teacher over-called ambiguous)
  + distribution mismatch. Fix = restore original intent labels / balance — but since
  spans are the priority, SPAN_MODE is shipped and intent is accepted as secondary.
- Oversampling a rare label that correlates with one intent (EXCLUDE≈creative) skews
  intent — don't expect intent balance and rare-oversample simultaneously.
- Empty-span docs (~11%, ambiguous support/gibberish) are **useful NER negatives** —
  they teach the model not to hallucinate spans; keep them.

---

## 7. Evaluation (span-focused; intent secondary)

Held-out hand gold (`gold_v2.json` + `gold_v2_extra.json`, ~71 queries, ≥15 per rare
label; leaked-into-train queries auto-excluded). Span metric = set-match per label:
`P=|P∩G|/|P|, R=|P∩G|/|G|, F=2PR/(P+R)`; report per-label + macro over priority labels.

**Final `model_v2_span` (macro span-F1 = 0.80):**
```
label           P     R     F1    n
CONCEPT        0.83  0.83  0.83   63
GIVEN_NAME     0.92  1.00  0.96   11
STYLE          0.92  0.79  0.85   14
REQUIRE_TOKEN  0.93  0.72  0.81   18
QUALIFIER      0.89  0.71  0.79   24
EXCLUDE_TOKEN  0.91  0.42  0.57   24     ← weakest recall (negation is hardest)
intent (secondary): 0.92 on this gold
```
Precision is high across all labels (0.83–0.93): when it fires, it's right. EXCLUDE
recall (0.42) is the known gap (negation scope on a small CNN) — backlog R5.
Also check
no concept/intent regression on the neutral hand-gold via the SAME eval code path.

---

## 8. Challenges & resolutions (all pattern-level, not per-query)
- spancat empty/noisy → tight-span teacher prompt + BILOU-NER.
- intent rule-overrides measured to HURT → intent = model alone.
- all-caps / extra whitespace mis-tokenized → `normalize_case` (case + whitespace collapse).
- TLD hardcoded .com → any leading-dot extension (regex).
- semantic fields on brittle regex/parser (qualifier, negation) → moved INTO the model
  as learned labels (v2 schema expansion) — regex kept ONLY for closed-form.
- multi-concept / coordinators / relative-clause objects / locations-as-concept /
  runaway GIVEN_NAME → fixed by teacher relabel + synthetic coverage.
- Remaining (backlog): EXCLUDE/REQUIRE recall on `the word X`/compound; comma-less
  lists; uncommon acronym brand-prefix — all data-side, near the small-model ceiling.

---

## 9. Why not a transformer (and where attention IS used)
`Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V`, cost ∝ n²·d — unbounded context, but:
| | CNN pipeline | BERT-base |
|---|---|---|
| context | ±8 tokens | full seq (n²) |
| params | few M | 110 M |
| CPU latency | ~11 ms | 30–100 ms |
| memory | ~120 MB | 250–500 MB |
| cost/req | $0 | wants GPU |
Measured: a distilled MiniLM encoder scored **0.80** intent vs the linear **0.82**, ~10×
slower — not worth it for short queries. Attention IS in the pipeline — as the **offline
teacher** (Claude), distilled into the CNN. Honest ceiling: long-distance/nested
negation is where attention would genuinely help; accepted limitation under the CPU/$0
mandate.

---

## 10. Companion docs
- `TRAINING_DATA_GUIDE.md` — how to create training data (the playbook).
- `TRAINING_DATA_BACKLOG.md` — remaining patterns to add (R1–R6).
- `NEXT_STEPS.md` — patterns A–I, model-optimization experiments (§3d), preprocessing notes.
- `OPERATIONS.md` — file execution flow, model storage, retraining & hosting (runbook).
