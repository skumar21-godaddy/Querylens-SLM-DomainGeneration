# QueryLens — Next Steps (improvement backlog)

> HISTORICAL NOTE: this backlog was written before the v2 rebuild. Its schema
> expansion (§1b: QUALIFIER/REQUIRE_TOKEN/EXCLUDE_TOKEN) and patterns A–G are now
> IMPLEMENTED in the shipped `model_v2_span`. The CURRENT remaining work is in
> `TRAINING_DATA_BACKLOG.md` (R1–R6). Kept here for the rationale/evidence trail.

Status (at time of writing): the systematic sweep over 400 unseen queries (78% clean)
+ targeted isolation tests
identified the remaining failure PATTERNS below. Each is a **data-coverage gap**,
fixed by adding teacher-labeled training examples — not by per-query rules.

---

## 1. Patterns to fix (evidence-backed, from `sweep_report.json` + isolation tests)

### Pattern A — Alternative coordinators drop the coordinated concept
**Symptom:** `engineering consulting and tax advisory along with music lessons`
→ misses `music lessons`.
**Root cause (isolated):** the model learned `and` / comma coordination but not
`along with` / `as well as` / `plus`. Training frequency: ` and ` = 1,297 rows,
`as well as` = 9, `along with` = 1.
**Fix:** add examples using non-`and` coordinators, all concepts labeled.

Training examples to add (intent = creative):
```json
{"text":"engineering consulting and tax advisory along with music lessons","intent":"creative","spans":[{"label":"CONCEPT","text":"engineering consulting"},{"label":"CONCEPT","text":"tax advisory"},{"label":"CONCEPT","text":"music lessons"}]}
{"text":"a bakery as well as a coffee roastery","intent":"creative","spans":[{"label":"CONCEPT","text":"bakery"},{"label":"CONCEPT","text":"coffee roastery"}]}
{"text":"pet grooming plus dog walking services","intent":"creative","spans":[{"label":"CONCEPT","text":"pet grooming"},{"label":"CONCEPT","text":"dog walking"}]}
{"text":"yoga studio together with a wellness cafe","intent":"creative","spans":[{"label":"CONCEPT","text":"yoga studio"},{"label":"CONCEPT","text":"wellness cafe"}]}
{"text":"web design & branding & seo","intent":"creative","spans":[{"label":"CONCEPT","text":"web design"},{"label":"CONCEPT","text":"branding"},{"label":"CONCEPT","text":"seo"}]}
```

### Pattern B — Concept buried in a relative clause is missed
**Symptom:** `a website that displays my event photography and videos` → only
`artistic website`; `Any galaxy name that is not taken` → nothing;
`a website for tracking surrogacy` → nothing.
**Root cause:** concepts that are OBJECTS of a verb inside a `that/which/who` clause
were pruned by the old teacher; the model never learned to extract them.
**Fix:** add relative-clause examples with the clause-internal object labeled.

Training examples to add:
```json
{"text":"a website that displays my event photography and videos","intent":"creative","spans":[{"label":"CONCEPT","text":"event photography"},{"label":"CONCEPT","text":"videos"}]}
{"text":"a platform that connects tutors with students","intent":"creative","spans":[{"label":"CONCEPT","text":"tutors"},{"label":"CONCEPT","text":"students"}]}
{"text":"an app which tracks surrogacy journeys","intent":"creative","spans":[{"label":"CONCEPT","text":"surrogacy"}]}
{"text":"a store that sells vintage records","intent":"creative","spans":[{"label":"CONCEPT","text":"vintage records"}]}
{"text":"any galaxy name that is not taken","intent":"creative","spans":[{"label":"CONCEPT","text":"galaxy name"}]}
```

### Pattern C — Descriptive phrase mislabeled as a given-name
**Symptom:** `Drone services south carolina`, `posture corrector improve posture`
→ tagged GIVEN_NAME instead of concept(+qualifier). ~biggest real-error bucket.
**Root cause:** the exact-file taught "Title-Case / short phrase = a name", so
Title-Case *descriptions* get mislabeled. Case is a real signal but over-weighted.
**Fix:** add Title-Case DESCRIPTIVE queries labeled as concepts (not names), plus
matching lower-case twins so case alone doesn't decide.

Training examples to add:
```json
{"text":"Drone Services South Carolina","intent":"creative","spans":[{"label":"CONCEPT","text":"Drone Services"}]}
{"text":"Posture Corrector For Desk Workers","intent":"creative","spans":[{"label":"CONCEPT","text":"Posture Corrector"}]}
{"text":"Mobile Car Detailing Business","intent":"creative","spans":[{"label":"CONCEPT","text":"Mobile Car Detailing"}]}
{"text":"Organic Skincare For Sensitive Skin","intent":"creative","spans":[{"label":"CONCEPT","text":"Organic Skincare"}]}
```
(Contrast — keep true names as GIVEN_NAME so we don't overcorrect:)
```json
{"text":"Two Gals and the Deadbeats","intent":"exact","spans":[{"label":"GIVEN_NAME","text":"Two Gals and the Deadbeats"}]}
```

---

### Pattern E — Short single-word concept + rare style in the "for X" template
**Symptom:** `suggest a dominant domain for the gym` → concept `[]`, style `[]`.
**Root cause (diagnosed):** NOT missing data — there are 261 "domain/name for X"
training rows, all with concepts, and 43% of concept spans are single-word. The
model (small CNN) simply under-confident on a *bare single common noun* ("gym") at
the end of the "for the X" template, where training mostly had multi-word concepts
("food product", "ai research"). Also STYLE recall is limited to frequent adjectives
(`fun`, `modern`); rare ones (`dominant`, `edgy`, `regal`) are missed.
**Fix — TRAINING ONLY (do NOT add a concept rule/fallback; that is cherry-picking):**
add examples of the template with single-word concepts and a wider STYLE vocabulary,
so the model *learns* it and generalizes.
```json
{"text":"suggest a dominant domain for the gym","intent":"creative","spans":[{"label":"CONCEPT","text":"gym"},{"label":"STYLE","text":"dominant"}]}
{"text":"a name for the bakery","intent":"creative","spans":[{"label":"CONCEPT","text":"bakery"}]}
{"text":"a domain for my clinic","intent":"creative","spans":[{"label":"CONCEPT","text":"clinic"}]}
{"text":"an edgy punchy name for a skate shop","intent":"creative","spans":[{"label":"CONCEPT","text":"skate shop"},{"label":"STYLE","text":"edgy"},{"label":"STYLE","text":"punchy"}]}
```

### Pattern F — Location tagged as CONCEPT (should be a qualifier, not a concept)
**Symptom (raw model):** `flower shop in canada` → CONCEPT `[flower shop, canada]`;
`… in maryland` → CONCEPT includes `maryland`. The model treats the location as a
concept. (A parser rule currently *also* flags it as a qualifier, causing the
duplicate — but the root fix is the model labeling.)
**Fix — TRAINING:** label locations/audiences as their own span type or leave them
OUT of CONCEPT in the training data, consistently.
```json
{"text":"flower shop in canada","intent":"creative","spans":[{"label":"CONCEPT","text":"flower shop"}]}
{"text":"web match making date sites in maryland","intent":"creative","spans":[{"label":"CONCEPT","text":"match making"},{"label":"CONCEPT","text":"date sites"}]}
```

### Pattern G — Runaway GIVEN_NAME on descriptive / aspirational sentences
**Symptom (raw model):** `Mountains are my dream and I want to show the world what it
is to be there` → GIVEN_NAME = the whole clause. The model over-extends GIVEN_NAME
across a full sentence that is a *description*, not a name; also mislabels domains
(`forkliftsunltd.com` → TLD) and short names as concept (`clean chimneys`).
**Fix — TRAINING:** add descriptive/aspirational sentences labeled with a tight
CONCEPT (or nothing), and clear GIVEN_NAME examples that are short, so the model
learns GIVEN_NAME is a short proper name, not a clause.
```json
{"text":"Mountains are my dream and I want to show the world what it is to be there","intent":"creative","spans":[{"label":"CONCEPT","text":"mountains"}]}
{"text":"forkliftsunltd.com","intent":"exact","spans":[{"label":"GIVEN_NAME","text":"forkliftsunltd.com"}]}
{"text":"i have a company named clean chimneys, suggest a domain","intent":"ambiguous","spans":[{"label":"GIVEN_NAME","text":"clean chimneys"}]}
```
**Sub-case — model grabs the "name" PREAMBLE, not the actual name.** On
`The name of the organization shall be Dignity Moves Foundation, a nonprofit …`
the model tagged GIVEN_NAME = `"The name of the organization"` (the lexical cue)
and MISSED `Dignity Moves Foundation`. Add "the name ... shall be/is X" examples so
the model learns the name is the proper noun AFTER the cue, not the cue itself:
```json
{"text":"the name of the organization shall be Dignity Moves Foundation, a nonprofit for foster children","intent":"exact","spans":[{"label":"GIVEN_NAME","text":"Dignity Moves Foundation"},{"label":"CONCEPT","text":"foster children"}]}
{"text":"my business name is Cedar & Sage Apothecary","intent":"exact","spans":[{"label":"GIVEN_NAME","text":"Cedar & Sage Apothecary"}]}
```

## 1b. ARCHITECTURAL redesign — move semantic extraction OFF rules, INTO the model

The brittle failures (regex negation broken by `not to have any ai word in result`;
parser tagging `result` as a qualifier) are because we put *context-dependent*
extraction on regex/parser. Fix = expand the NER label schema so the model LEARNS
them from context; keep regex only for closed-form/numeric constraints.

**New NER labels to add (teacher-labeled, jointly trained):**
- `QUALIFIER` — audience/location/topic ("kids", "canada"). Replaces the parser
  `_qualifiers` rule (which fires on any `in X`, e.g. wrongly on "in result").
- `REQUIRE_TOKEN` / `EXCLUDE_TOKEN` — a token the name must / must-not contain. The
  model learns the exclude sense from surrounding negation ("not to have any ai") —
  something regex cannot do at arbitrary distance. Replaces the negation regex.

**Stays regex (genuinely closed-form):** `length_range`, `tld`, `no_hyphen`,
`no_digits`, `price_max`. A model adds only noise to numeric/structural extraction.

**Teacher-labeling examples for the new labels:**
```json
{"text":"suggest a domain for data engineering but not to have any ai word in the result","intent":"creative","spans":[{"label":"CONCEPT","text":"data engineering"},{"label":"EXCLUDE_TOKEN","text":"ai"}]}
{"text":"a name that must include the word cloud","intent":"creative","spans":[{"label":"REQUIRE_TOKEN","text":"cloud"}]}
{"text":"nootropic supplement for kids and teens","intent":"creative","spans":[{"label":"CONCEPT","text":"nootropic supplement"},{"label":"QUALIFIER","text":"kids and teens"}]}
{"text":"flower shop in canada","intent":"creative","spans":[{"label":"CONCEPT","text":"flower shop"},{"label":"QUALIFIER","text":"canada"}]}
```

**Honest ceiling:** the serving model is a small CNN (±8-token window). It will beat
regex/parser decisively on qualifiers + negation, but long-distance / nested negation
is where self-attention (a transformer) or the offline LLM is genuinely better.
Within the CPU-only, $0/query mandate this schema is the right call; adversarial
compositional negation remains an accepted, documented limitation.

### Pattern H — Comma-less / unpunctuated coordinated lists (near the ceiling)
**Symptom:** `portfolio creation for barbers nail artists massage threpaist and anybody
who works in beauty …` → the parser merges `nail artists massage threpaist` into ONE
chunk, so `nail artists` is lost and professions get mixed between concept/qualifier.
**Root cause:** no delimiter between list items; segmenting them needs world knowledge
("nail artists" vs "massage therapist" are distinct), compounded by misspellings
(`threpaist`) and 40-word run-on text.
**Fix — TRAINING (partial):** with the schema expansion the professions become
`QUALIFIER` (audience) instead of mixed concepts, and adding comma-less-list examples
teaches common segmentations. **Honest limit:** arbitrary comma-less lists + typos are
where the small CNN caps out; only a larger/attention model or the LLM reads them
cleanly. Document as a known limitation, do NOT chase with rules.
```json
{"text":"portfolio creation for barbers nail artists and massage therapists","intent":"creative","spans":[{"label":"CONCEPT","text":"portfolio creation"},{"label":"QUALIFIER","text":"barbers"},{"label":"QUALIFIER","text":"nail artists"},{"label":"QUALIFIER","text":"massage therapists"}]}
```

## 2. Rule-side guard (no retrain) — Pattern D: over-fire on long/garbled queries
**Symptom:** rambling multi-sentence input yields junk `required_token` ("much
deeper") and over-long qualifiers.
**Fix (in `intent_layer/constraints.py` + `hybrid._qualifiers`):**
- skip `required_token` / qualifier matches whose captured phrase is > 3 words or
  contains sentence punctuation;
- cap qualifier phrase length and drop if it spans a clause boundary (`—`, `.`, `;`).
These are precision guards on existing rules, general (length/punctuation based),
not per-query.

---

## 3. Execution plan for the next relabel + retrain round
1. **Assemble candidates** (query-selection over the source CSVs):
   - existing under-coverage detector (content-NPs ≫ concepts), PLUS
   - queries containing alt-coordinators (`along with|as well as|plus|together with|&`), PLUS
   - queries with a relative clause (`\b(that|which|who)\s+\w+`), PLUS
   - Title-Case descriptive queries (from the exact file) with no domain suffix.
   Target ~600–800 fresh candidates from BOTH files.
   - **NEW data source:** `6plus_word_queries.csv` (2,498 rows, median 10 words) — the
     hard long/conversational subset with purchase columns. Prime candidate pool for
     Patterns A/B/F/G/H; also usable later for offline purchase-impact signal.
2. **Relabel** with `relabel_mc.py` (strict "extract EVERY concept" prompt; already
   covers coordinated lists — extend the prompt with an `along with`/`as well as`
   worked example and a Title-Case-description example).
3. **Add the hand-written seed examples** above (Patterns A–C) to guarantee coverage
   of rare connectives the sample may under-represent.
4. **Patch + build** (`make_training_v2.py`) and **retrain** (`config_ner.cfg` → `model_v2_span`).
5. **Evaluate** (`eval_v2.py` + a QA sweep over unseen queries): require secondary-concept recall and
   full-MC coverage to rise, intent flat on neutral hand-gold, and the sweep's
   `under_coverage`/`relative_clause` buckets to shrink — without new `long_concept`
   or `concept_qualifier_overlap` flags.
6. **Ship** only if the neutral metrics hold (same honesty bar as before).

---

## 1c. Synthetic data augmentation (for rare/hard patterns only)

Some patterns are too RARE in real data for the model to learn (aspirational/poetic
briefs like "Mountains are my dream and I want to show the world…"; `along with`
coordinators; complex negation). Use the LLM teacher to GENERATE + label queries in
these styles to fill the gap.

**Rules to keep it safe (synthetic data hurts if naive):**
- **Seed from real queries.** Paraphrase real rows (`6plus_word_queries.csv`) into the
  target style rather than inventing from scratch — keeps phrasing realistic; inject
  typos/run-ons on purpose so the model stays robust to messy input.
- **Cap the proportion** (≤15–20% of the training set). Over-injection skews the
  distribution and regresses common cases (cf. the failed file-remix experiment).
- **Validate ONLY on real held-out data** (sweep + hand-gold), never on synthetic —
  avoids the self-consistency trap (same LLM writing and labeling).
- **Label honestly.** For aspirational queries with no concrete concept, the target is
  a TIGHT concept (e.g. `mountains`) or none — and explicitly NOT a runaway GIVEN_NAME.
```json
{"text":"Mountains are my dream and I want to show the world what it is to be there","intent":"creative","spans":[{"label":"CONCEPT","text":"mountains"}]}
{"text":"I want to inspire busy parents to slow down and reconnect","intent":"creative","spans":[{"label":"CONCEPT","text":"busy parents"}]}
```

## 3b. Preprocessing / robustness notes (from raw-mode probing)
- **Train/serve case mismatch (fix in retrain):** `normalize_case` (all-caps→lower)
  runs at SERVE time (`hybrid`) but the DocBin builders use RAW text. Apply
  `normalize_case` inside `make_training_*` so train == serve distribution.
- **No stopword removal / lemmatization — keep it that way.** NER+textcat need the
  full raw token sequence; stripping stopwords would destroy boundary/intent signal.
- **Do NOT spell-correct before the model.** Domain queries contain intentional
  non-words (brands, invented names — e.g. "Vexa, Kyro, Zyn"); a corrector would
  corrupt them. Handle typos via (a) tok2vec PREFIX/SUFFIX/SHAPE features (already on)
  and (b) training on real typo-laden queries (`6plus_word_queries.csv`). Any future
  correction must be surgical (lowercase, non-brand, 1-edit from a common word only).
- **`invented words like X` → STYLE, not REQUIRE_TOKEN**; abstract quality words
  (`excellence`, `integrity`) → keep OUT of CONCEPT. Both handled by the learned
  REQUIRE_TOKEN/STYLE labels + cleaner concept labeling in the relabel round.

## 3c. Labeling-convention refinements (next relabel pass)
- **Short standalone name-like phrases** ("happy era", "see you in the underground")
  currently get intent=ambiguous with EMPTY spans. Convention going forward: tag them
  `GIVEN_NAME` (the user typed a candidate/name), even when intent stays ambiguous.
- **Support/account queries** ("set up email", "domain hosting price") — keep as
  ambiguous+empty (useful negatives so the model learns NOT to extract), OR filter
  upstream. Decided: keep for now (teaches non-response), revisit later.
- **Candidate-name lists** ("Qircle, Qanban, Qanvas, …") → label each as GIVEN_NAME.

## 3d. Model optimization (measured, one lever at a time)
Diagnostic finding (v2): intent regressed 0.80→0.68 on neutral hand-gold, while spans
IMPROVED (CONCEPT 0.85, QUALIFIER 0.83). Teacher intent ceiling = 0.82; mcfix reached
0.80, v2 only 0.68 → the model is UNDER-learning intent (not a label problem). Cause:
the richer 6-label NER task steals capacity from the tok2vec SHARED with textcat.

Ordered experiments (keep a change only if new-label gold improves AND concept/intent
do NOT regress on neutral gold):
1. **Decouple textcat encoder (IN PROGRESS)** — give intent its own tok2vec so it stops
   competing with the 6-label NER. `config_v2_dec.cfg` → `model_v2_dec`. Expect intent →~0.80.
2. **Better vectors** — en_core_web_lg or floret subword (OOV/typo robustness). High value.
3. **Dropout 0.1→0.2–0.3** — reduce overfit on ~5k data; protect good cases.
4. **NER hidden_width 64→128** — more capacity for the 6-label span task (lift EXCLUDE).
Not recommended: wider/deeper receptive field (overfits on 5k). Transformer: offline
challenger only (breaks CPU/$0 mandate).

### Pattern I — Uncommon acronym brand-prefix in a description (near ceiling)
**Symptom:** "a VI media collaboration company …" → concept `media collaboration`,
drops `VI`. The model DOES capture it with a naming cue ("company called VI Media" →
GIVEN_NAME) and keeps common acronyms ("AI media company"). Bare uncommon 2-letter
acronyms (`VI`) with no cue are genuinely ambiguous (brand vs Roman numeral vs filler).
**Fix — TRAINING (marginal), do NOT rule-patch** (a rule over-fires, cf. "3M style
brand for adhesives" wrongly captured whole as GIVEN_NAME). Add examples of uncommon
acronym brand-prefixes in descriptions. Accepted limitation: bare no-cue acronyms stay imperfect.

## 4. Open items (separate from the relabel round)
- **Intent override for "named/called X + request → ambiguous":** now that given-name
  extraction is reliable, MEASURE this single high-precision override on blind data
  before enabling (broad overrides previously measured net-negative: 64.4% vs 72.3%).
- **Fairly-labeled precision gold:** current concept precision is understated because
  `consensus_gold` under-labels concepts. Build a small exhaustively-relabeled
  multi-concept gold slice to measure precision without the artifact.
- **Online purchase impact:** all results are on non-purchase data. The purchase lift
  vs PRAG is a guardrailed Hivemind A/B, not an offline claim.
- **Re-publish Confluence** page 4595417640 with the final `model_v2_span`
  numbers + a known-limitations section once the round lands.
