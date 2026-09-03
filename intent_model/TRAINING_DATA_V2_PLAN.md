# QueryLens — Training Data v2 Plan (schema-expanded relabel + retrain)

Confirmed decisions: re-label ~5,000 diverse existing rows with the new schema;
add ~1,200 real (6plus) + ~800 synthetic; build a ~150-query gold for the new labels.
Total training ≈ 7,000. Goal: move semantic extraction OFF rules INTO the model.

---

## 1. Schema (the core change)

**Intent (textcat, unchanged, 3-class):** `creative` | `exact` | `ambiguous`

**NER span labels (6) — LEARNED by the model:**
| label | definition | tight? |
|---|---|---|
| `CONCEPT` | the subject/product/theme/deliverable being named. Every distinct one. NOT audience/location. | 1–3 words each |
| `STYLE` | adjective describing the desired NAME (`fun, modern, premium, minimalist, invented, brandable, edgy`) | 1 word |
| `GIVEN_NAME` | a specific name the user ALREADY has — the actual proper noun only (NOT the preamble "the name of the org"); include domains (`forkliftsunltd.com`) and suffixes (`Acme LLC`) | tight |
| `QUALIFIER` | audience / location / target-market (`kids`, `canada`, `barbers`, `small businesses`) — who/where, NOT the concept | tight |
| `REQUIRE_TOKEN` | a word the NAME must contain (`the word cloud`, `must include fresh`) — the token itself | 1–2 words |
| `EXCLUDE_TOKEN` | a word the name must NOT contain (`not contain ai`, `avoid watch/time/chrono`, `without crypto`) — the token itself | 1–2 words |

**Closed-form, REGEX-only (NOT learned):** `tld`, `length_range`, `no_hyphen`,
`no_digits`, `price_max` / `price_min`, `position` (start/end).

**Critical disambiguation rules for the teacher prompt:**
- CONCEPT vs QUALIFIER: the thing being named = CONCEPT; who/where it serves = QUALIFIER.
  `supplement for kids` → CONCEPT `supplement`, QUALIFIER `kids`.
- REQUIRE vs EXCLUDE: decided by negation in context (`must include X` vs `not/avoid/without X`).
- GIVEN_NAME is the actual name only — `the name shall be Dignity Moves Foundation` →
  GIVEN_NAME `Dignity Moves Foundation` (NOT "the name"). Aspirational sentences get a
  TIGHT concept or none — never a whole-clause GIVEN_NAME.
- `invented words like X`, `premium`, `brandable` → STYLE, never REQUIRE_TOKEN.
- Abstract quality words (`excellence`, `integrity`) → NOT concepts.

---

## 2. Data composition (≈7,000)

| bucket | count | source | selection |
|---|---|---|---|
| relabel existing | ~5,000 | `claude_labels_v2` + `creative_extra` (7,336 pool) | diverse sample: keep intent balance + length spread; **full re-label** with new schema |
| new real | ~1,200 | `6plus_word_queries.csv` ONLY | unique, ≥10–12 words, novel (not already labeled). 1,137 available ≥10w; top up from ≥12w |
| synthetic | ~800 | Opus-generated | 20–35 words, rare-label coverage, seeded from real phrasing, some typos/run-ons |

Length bias: prefer longer queries; **cap the 100+ word monsters** (represent, don't
over-weight). Dedup across all sources + against existing labels (normalized text).

---

## 3. Synthetic generation spec (Opus, offline)

~800 queries, **20–35 words**, deliberately covering the labels real data under-supplies:
- QUALIFIER-heavy (audiences/locations, incl. comma-less profession lists)
- REQUIRE_TOKEN / EXCLUDE_TOKEN with **varied negation phrasing** (`not`, `without`, `avoid`, `no`, distant negation)
- aspirational / emotional briefs (tight concept or none)
- GIVEN_NAME constructions (`company named X`, `name shall be X`, `my domain is X.com`)
- coordinated lists with `along with` / `as well as` / `plus`
Seed from real 6plus phrasing; inject occasional typos. Cap ≈11% of total (drift guard).
Generated AND labeled by Opus in one strict pass; **validated only on real gold**.

---

## 3.5 Pattern-coverage matrix (guarantee every NEXT_STEPS pattern is trained)

Candidate selection + synthetic generation MUST each explicitly hit all of these
(else the round under-covers them). Real = filter on the 3 files; Synth = Opus bucket.

| # | pattern (NEXT_STEPS) | how covered |
|---|---|---|
| A | alt coordinators (`along with`/`as well as`/`plus`) | Real: regex-select queries w/ these connectives; Synth bucket |
| B | concept in relative clause (`that/which/who` + verb) | Real: regex-select `\b(that\|which\|who)\s+\w+`; Synth bucket |
| C | Title-Case descriptive → not a name | Real: Title-Case queries from exact-file w/ no domain suffix; Synth twins (lower+Title) |
| E | single-word concept in `name/domain for X` + rare STYLE | Real: `(name\|domain) for (a\|the) <1 noun>`; Synth w/ varied STYLE adjectives |
| F | location tagged as concept → QUALIFIER | via schema (QUALIFIER label) + Synth location/audience bucket |
| G | runaway GIVEN_NAME / aspirational / `name shall be X` | Synth aspirational + given-name-construction buckets |
| H | comma-less coordinated lists (audience) | Real: profession/item lists; Synth comma-less bucket (accepted-ceiling) |
| D | required-token over-fire on long/garbled input | resolved by moving REQUIRE/EXCLUDE to the model (schema) |

Deferred (NEXT_STEPS §4, NOT this round): intent-override measurement
(named+request→ambiguous), fair-precision gold, online purchase A/B, Confluence
re-publish. Called out so they aren't silently dropped.

## 4. Teacher prompt (the make-or-break artifact)
One strict SYSTEM prompt used for BOTH relabel and synthetic-labeling, emitting the
6-label schema with the disambiguation rules above, tight spans, verbatim substrings,
worked examples per label (esp. CONCEPT-vs-QUALIFIER and REQUIRE-vs-EXCLUDE). Batched,
checkpointed/resumable (like `relabel_mc.py`).

---

## 5. Build → retrain → refactor → validate
1. Build DocBin with the 6 NER labels + intent one-hot (extend `make_training_*`, apply
   `normalize_case` at train time to fix the train/serve mismatch).
2. Retrain `config_ner.cfg` → `model_v2`.
3. **Refactor serving** (`hybrid.py`): qualifiers from MODEL (drop parser `_qualifiers`);
   require/exclude from MODEL (drop regex + brittle negation); regex keeps tld/length/
   hyphen/digit/price/position only. Extend price regex to `price_min`.
4. **Validate:**
   - new ~150-query gold (hand/consensus) → P/R for QUALIFIER/REQUIRE/EXCLUDE/GIVEN_NAME
   - existing consensus_gold + hand-gold → CONCEPT + intent must NOT regress
   - a QA sweep over unseen queries → flagged buckets (under_coverage, empty_all) must shrink
5. **Ship `model_v2`** only if new-label P/R is solid AND concept/intent hold on neutral gold.

## 6. Success criteria (honest bar)
- QUALIFIER / REQUIRE_TOKEN / EXCLUDE_TOKEN reach usable P/R on the new gold (≥~0.6 F1).
- CONCEPT recall + intent do NOT regress on hand-gold vs current `model_mcfix_full`.
- Sweep `empty_all` + `under_coverage` shrink; no new `long_concept` blowups.
- Accepted limitation: adversarial long-distance negation / comma-less lists remain imperfect.
