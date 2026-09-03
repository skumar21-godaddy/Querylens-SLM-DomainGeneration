# QueryLens — Training-Data Creation Guide

A complete playbook for anyone retraining the GoDaddy domain-search intent+span model.
Read this end-to-end before generating labels. Every rule here was learned the hard way
(the "Why it matters" notes are real failures we hit). The golden rule:

> **The model is only as good as its labels. Data quality/consistency beats
> hyperparameters and beats model size on this task. Spend your effort here.**

---

## 0. What the model is
A CPU-only spaCy pipeline: `tok2vec (CNN) → textcat (intent) → ner (spans)`, trained by
DISTILLING a large LLM (Claude, offline) that labels queries. The small model serves at
~10 ms, $0/query; the LLM never runs in production. So "creating training data" =
running the LLM teacher over well-chosen queries with a strict schema, plus synthetic
augmentation, then building a balanced DocBin.

---

## 1. The label schema (fixed — keep it consistent)

**Intent (textcat, exclusive):** `creative` | `exact` | `ambiguous`.
(Open option: collapse to `exact` | `creative`. If you do, do it in the teacher prompt
AND everywhere, and rebuild the whole set — never mix schemas.)

**Span labels the MODEL learns (NER):**
| label | what it is | keep it |
|---|---|---|
| CONCEPT | subject/product/theme/deliverable being named. EVERY distinct one. | 1–3 words each |
| STYLE | adjective for the NAME (fun, modern, premium, invented, minimalist) | 1 word |
| GIVEN_NAME | a name the user ALREADY has — the ACTUAL proper name only | short; incl. domains, LLC |
| QUALIFIER | audience / location / target-market (kids, canada, barbers) | tight |
| REQUIRE_TOKEN | a word the name MUST contain | 1–2 words |
| EXCLUDE_TOKEN | a word the name must NOT contain (negation) | 1–2 words |

**Handled by REGEX, NOT labeled (do not tag these):** TLD (.com/.io), length
("4-12 chars"), price ("under $800"), no-hyphen/no-digit, start/end position.

**Golden disambiguation rules (the ones that cause label noise if wrong):**
1. CONCEPT vs QUALIFIER — the thing being named = CONCEPT; who/where it serves =
   QUALIFIER. "supplement for kids" → CONCEPT `supplement`, QUALIFIER `kids`.
2. REQUIRE vs EXCLUDE — read the negation from context, even several words away
   ("not to have any ai word" → EXCLUDE `ai`).
3. GIVEN_NAME = the actual name only. "the name shall be Dignity Moves Foundation" →
   `Dignity Moves Foundation`, NOT "the name". Never a whole descriptive clause.
4. `invented words like X`, `premium`, `brandable` → STYLE, never a required token.
5. Abstract fluff (excellence, integrity, passion) → NOT a concept.
6. Vague/aspirational sentence with no concrete subject → a TIGHT concept or nothing;
   NEVER a runaway GIVEN_NAME across the sentence.

> **Why it matters:** the biggest measured regressions came from label *inconsistency*
> (e.g. the teacher over-calling "ambiguous", or dropping the 2nd concept). One
> ambiguous rule applied inconsistently teaches the model the wrong pattern.

---

## 2. Choosing the queries (sources + selection)

**Sources:** `nonpurchase_creative.csv`, `nonpurchase_exact_entity.csv`,
`6plus_word_queries.csv` (long conversational; has purchase columns). Only English.

**Rules:**
- **Dedup by NORMALISED text** (`re.sub(r"\s+"," ",q.lower()).strip()`). `6plus` has
  ~18% internal duplicates — always dedup first.
- **Bias toward longer queries** (≥10–12 words) for NEW selections — that's where the
  hard patterns (multi-concept, qualifiers, negation) live and where the model is
  weakest. But **don't over-weight the 100+ word monsters** (represent, not dominate).
- **Novelty:** new real queries should NOT already be in the labeled set (check against
  existing labels). Use `6plus` for genuinely new long queries.
- Keep a broad **length spread** and industry spread, not just the long tail.

(Query-selection: stratified, length-biased, deduped — over the source CSVs.)

---

## 3. Running the teacher (label_v2.py)

- The SYSTEM prompt in `label_v2.py` IS the schema contract. Change the schema? Change
  the prompt (with worked examples per label) and re-label EVERYTHING.
- Batch ~15 queries/call; **checkpoint to a JSONL and make it resumable** (skip already
  -labeled texts) — a 7k-query run takes ~45–70 min and WILL get interrupted.
- Auth token from env `CAAS_AUTH_TOKEN` (never hardcode).
- **Validate the teacher before trusting it at scale:** label the ~90-query hand-gold
  and check intent agreement (expect ~0.82; that is your realistic ceiling — the student
  can't beat its teacher).

---

## 4. Synthetic augmentation (gen_synthetic_v2.py) — for RARE patterns only

Real data under-supplies some patterns (negation/exclude, aspirational, `along with`).
Generate them with the LLM. **Discipline (synthetic hurts if naive):**
- **Seed from real phrasing** (paraphrase real rows); inject typos/run-ons so the model
  stays robust to messy input. Don't produce unnaturally clean text.
- **Cap synthetic at ≤ ~12%** of the final set. Over-injecting a rare pattern skews the
  distribution and REGRESSES common cases (we measured this).
- Generate 20–35-word queries (standard length; don't over-complicate).
- **Validate only on REAL held-out gold**, never on synthetic (self-consistency trap:
  same LLM writing + labeling looks right but isn't).
- Label synthetic with the SAME teacher/prompt as real (schema consistency).

---

## 5. Building the DocBin (make_training_v2.py) — merge, balance, split

1. **Merge** real + synthetic; **dedup by normalised text** (real wins on collision).
2. **Cap synthetic share** (~12%).
3. **normalize_case at TRAIN time** (all-caps→lower, collapse whitespace) — MUST match
   serve-time or you get a train/serve mismatch. This lives in `LF.normalize_case`.
4. **Balancing — this is subtle, read carefully:**
   - **If optimizing SPANS (default priority):** keep ALL data (more span examples), do
     NOT downsample by intent, and OVERSAMPLE the weak rare labels (EXCLUDE ×3,
     REQUIRE ×2). `SPAN_MODE=1`. This gave the best span scores.
   - **If you need good intent too:** equal intent balance (≈33/33/33) recovers intent
     (~0.80 on hand-gold), but downsampling throws away span data and can lower span
     recall. `BALANCE=1`. It's a trade-off — pick by objective.
   - **Do NOT** oversample a rare label that correlates with one intent (EXCLUDE is
     creative-heavy) AND expect intent balance — they fight. We saw exclude-oversample
     skew intent toward creative.
5. **Empty-span docs are GOOD** (~11%): support/gibberish/vague queries labeled
   ambiguous+empty are NER *negatives* that teach the model NOT to hallucinate spans.
   Keep them. Do not force spans onto genuine junk.
6. **Span→offset mapping** (`_spans_no_overlap`): match span text case-insensitively,
   longest-first, drop overlaps (NER needs disjoint spans). Verify spans align to token
   boundaries (`alignment_mode="contract"`).
7. **Split** ~8% dev; keep the eval GOLD entirely separate (never in train/dev).

---

## 6. Evaluation (the honesty bar)

- **Hold out a hand-labeled gold** (`gold_v2.json` + `gold_v2_extra.json`) with enough
  examples PER rare label (≥15–20 each) — small samples (n=4–8) are too noisy to pick a
  model. Auto-exclude any gold query that leaked into training.
- Report **per-label P/R/F** and a **macro span-F1** over the priority labels. Rank
  models by span-F1 (intent is secondary).
- **Regression check** on the neutral hand-gold: a change must not drop concept/intent
  there, even if it helps elsewhere.
- **Run a QA sweep** over unseen queries; the flagged buckets
  (empty_all, under_coverage) should SHRINK.
- **Compare apples-to-apples:** eval every model through the SAME code path
  (same normalize_case). We once mis-attributed a "regression" to the wrong cause
  because baselines used different preprocessing.

---

## 7. Things that DON'T help (measured — don't repeat these)
- **Spell-correcting input** — domain queries are full of intentional non-words (brands,
  invented names "Vexa/Kyro"); a corrector corrupts them. Rely on tok2vec subword
  features + training on real typo-laden data instead.
- **Stopword removal / lemmatization** — NER+textcat need the full token sequence;
  stripping stopwords destroys "name FOR X" vs "name X".
- **Re-mixing file proportions** (e.g. 50/50) without fixing labels — measured net-negative.
- **Wider/deeper CNN receptive field** on ~5–7k data — overfits, regresses good cases.
- **Rules to patch a single failed query** — they don't generalize and rot the codebase.
  Fix the PATTERN in the data; use rules only for closed-form constraints (tld/length/price).

---

## 8. End-to-end checklist
```
[ ] dedup all sources (normalised); bias new picks to ≥10–12 words
[ ] confirm the teacher SYSTEM prompt matches the schema; validate on hand-gold (~0.82)
[ ] relabel ALL kept rows with the current schema (never mix old/new schema)
[ ] generate synthetic ONLY for rare patterns; seed from real; cap ≤12%; inject typos
[ ] merge + dedup (real wins) + cap synthetic
[ ] normalize_case at train time (case + whitespace) == serve
[ ] pick balancing by objective: SPAN_MODE (spans) or BALANCE (intent)
[ ] keep empty-ambiguous negatives; keep spans disjoint & token-aligned
[ ] hold out gold with ≥15–20 examples per rare label; no leakage
[ ] train; select best by macro span-F1; regression-check neutral gold; run sweep
[ ] refactor serving so learned labels come from the model, regex only for closed-form
```
