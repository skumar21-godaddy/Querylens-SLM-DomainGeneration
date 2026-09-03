# QueryLens — Training-Data Backlog (remaining patterns)

This lists the failure patterns captured from live testing that the **current v2 span
model does NOT yet fully handle**. It deliberately EXCLUDES patterns already covered
by the v2 schema + relabel + synthetic round (those are in the "Already covered"
section at the bottom for reference). Feed these into the *next* relabel/synthetic pass.

Priority: SPAN quality (concept, given_name, style, require_token, exclude_token,
qualifier) is the non-negotiable objective. Intent is secondary (creative/ambiguous
confusion is acceptable; a 2-intent exact/creative collapse is an open option).

---

## Open patterns (add training examples for these)

### R1 — Verb-PHRASE actions as concepts (extends Pattern B)
Concepts that are the ACTION a service performs, expressed as a verb phrase inside a
relative/participial clause, are dropped.
- "Global experts **that save** hotels from loss and **increase revenue**" → misses `save`/`increase`
- "a website **that hosts** videos" → (noun object "videos" now caught, but the action verb isn't)
Decision so far: extract the noun objects, not the verbs — but where the verb IS the
value proposition, consider a short CONCEPT ("revenue recovery", "loss prevention").
Add examples labeling the nominalised action as CONCEPT. **Marginal — verbs are usually
not the concept; only add where the action is clearly the offering.**

### R2 — Comma-less / unpunctuated coordinated lists (Pattern H, near ceiling)
"barbers nail artists massage therapist" (no delimiters) → items merge; "nail artists"
lost. Needs examples of delimiter-free profession/item lists (with typos). **Accepted
ceiling:** arbitrary comma-less lists + misspellings remain imperfect on a small CNN.

### R3 — Uncommon acronym brand-prefix in a description (Pattern I, near ceiling)
"a **VI** media collaboration company" → drops `VI`. Model captures it WITH a naming cue
("called VI Media") and keeps common acronyms ("AI media"). Add uncommon-acronym-prefix
examples. **Do NOT rule-patch** (over-fires, e.g. "3M style brand…" whole-captured).
**Accepted ceiling:** bare no-cue 2-letter acronyms stay ambiguous.

### R4 — Short standalone name-like phrases labeled empty (from §3c)
"happy era", "see you in the underground" → currently ambiguous + EMPTY. Convention:
tag as `GIVEN_NAME` (the user typed a candidate/name), even if intent stays ambiguous.
Candidate-name lists ("Qircle, Qanban, Qanvas…") → label EACH as GIVEN_NAME.

### R5 — EXCLUDE_TOKEN / REQUIRE_TOKEN recall (confirmed weak on final model)
Final v2 span model: REQUIRE F1 0.81 (R 0.72), EXCLUDE F1 0.57 (R 0.42) — precision is
high (0.91–0.93, no false fires) but recall lags. Specific failures observed:
- "using **the word** user" → require `user` missed; "must include **the word** sweet" missed.
  The `the word X` construction is under-learned (synthetic mostly used "must include X").
- COMPOUND require+exclude in one query ("must include sweet but avoid cake") → BOTH missed.
- COMPOUND exclude LIST ("dont include the name book or tees") → caught `tees`, dropped `book`
  (only one of two tokens). Add multi-token "X or Y/Z" exclude examples.
- INTERFERENCE: when an excluded token also appears earlier as a CONCEPT (e.g. "selling
  self-published books … dont include the name book") the model tags it CONCEPT, not
  EXCLUDE. Add examples where the same lemma is both a product concept and an excluded name-word.
- Fires correctly on direct forms ("no ai or tech" → exclude ai, tech).
**Fix:** add many more examples with (a) the `the word X` / `with the word X` phrasing,
(b) compound require+exclude in one query, (c) common-word tokens (user, used), (d)
distant negation. Mix real + synthetic; these are the hardest labels (negation scope on
a small CNN) — set expectations that exclude recall stays the lowest.

### R6 — Distant / subordinate multi-concept recall
Long briefs still under-recall the 3rd/4th concept when it's far from the head or in a
subordinate clause. Add long multi-concept examples (the `6plus_word_queries.csv` pool).

---

## Non-data / related items
- **Model optimization** (see NEXT_STEPS §3d): floret/lg vectors for OOV+typo robustness;
  NER hidden_width 64→128 for the 6-label task. Measure one at a time.
- **Fair-precision gold:** `consensus_gold` under-labels concepts, so concept PRECISION
  is understated. Build an exhaustively-relabeled slice to measure precision cleanly.
- **Intent 2-class collapse** (exact/creative) — open product decision; would simplify
  the head and remove the ambiguous confusion entirely.
- **Online purchase A/B** — all metrics are non-purchase; validate lift vs PRAG live.

---

## Already covered by the v2 round (do NOT redo)
- Schema expansion: QUALIFIER, REQUIRE_TOKEN, EXCLUDE_TOKEN now learned by the model.
- A — alternative coordinators (along with / as well as / plus).
- B (noun objects) — concepts as objects in "that/which" clauses (event photography, videos).
- C — Title-Case descriptive phrases → concept, not given-name.
- E — single-word concept in "name/domain for X" + rare STYLE adjectives.
- F — location/audience → QUALIFIER (not concept).
- G — aspirational sentences (tight concept, no runaway GIVEN_NAME); "name shall be X"
  → the actual name; referenced domains → GIVEN_NAME.
- Whitespace: multiple spaces collapsed in `normalize_case` (train+serve).
- Preprocessing decisions: NO stopword removal, NO spell-correction (see guide).
