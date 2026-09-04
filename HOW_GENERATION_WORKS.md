# How Domain Generation Works

This file explains, in two parts, exactly what happens when you type a query and get
10 domains back:

- **Part 1 — Plain English.** What the model is asked to do, what *we* guarantee with our
  own code, and who is responsible for what. No code.
- **Part 2 — The flow.** How the query becomes a brief, how the brief becomes a prompt,
  what gets sent to the SLM, and how the answer is checked before you see it.

The whole pipeline is: **your query → QueryLens brief → simple description → SLM → hard-constraint checks → 10 domains.**

---

## Part 1 — Plain English

### The design principle
A small language model (SLM) is genuinely good at two things: understanding a short
description, and inventing brandable names. It is *not* reliable at obeying a long list of
rules. So the split is deliberate:

- **The model owns meaning and creativity** — reading what you want, and naming it.
- **Code owns only the handful of things that must be exactly true** — the TLD, a length
  limit, banned words, and so on.

An earlier version of this project tried to *force* good names with code — inject the key word
into N names, synthesize combos, keep word-lists of "bad" terms. It worked for the one query it
was tuned on and failed to generalize (it produced keyword-stuffed names like
`registryreformregistry`). We removed all of that. The lesson: **don't out-think the model with
rules; give it a cleaner input and let it name.**

### The two prompts, working together
Generation uses two messages, and they're designed as a pair:

- **The system prompt is the "how to name well" instruction, written once, for every query.**
  It tells the model: sound like a real brand (don't just glue the words together), keep names
  short and lowercase, capture the *one* core idea (don't copy long phrases or list every
  detail), skip generic filler like "nonprofit"/"company", never use stigmatizing words, and
  reply as JSON.

- **The user message is a short, plain description of *what* to name** — the way you'd brief a
  designer in one or two sentences. It changes by intent:
  - **Exact** ("The Lounge Coffee Co") — "the business is already named X; give close
    variations."
  - **Name + concepts** ("our server *bean*, used for minecraft/dns/jellyfin") — "keep *bean*
    in every name and pair it with one short related word."
  - **Creative** ("a fun vegan bakery for kids") — "Describe: a vegan bakery. It should feel
    fun. It is for kids." Then let the model name it.

Because the system prompt already carries the quality bar, the per-query message stays simple.
We deliberately **do not** tell the model "use word X in 5 names" — that kind of instruction
makes a 3B *fixate* and stuff one word everywhere. Letting it read the description is what keeps
the names varied and on-idea.

### What the SLM generates, and what we guarantee
The SLM returns ~10 candidate names. It's creative but imperfect: it sometimes forgets the dot
before the TLD, picks an odd extension, ignores a length limit, uses a banned word, or repeats
itself. So a small, boring code step (`enforce`) fixes **only** the things that must be exactly
true:

| What code guarantees (hard, checkable)                         | Why it's not left to the model |
|----------------------------------------------------------------|--------------------------------|
| **Exact-match first** — if you already have a name, results 1–2 are that exact name (`.com`/`.co`) | the model paraphrases it away |
| **Required TLD / at least 6 `.com`**                           | the model scatters extensions |
| **Sensible extensions** (no `.name`/`.tech` on a bakery)       | the model picks irrelevant TLDs |
| **Length limit, no-hyphen, no-digits**                         | the model ignores numeric limits |
| **Must-include / must-avoid words** (`require`/`exclude`)      | the model uses a banned word |
| **Safety: drop stigmatizing names** (`offender`, `svp`, …)     | the prompt asks it to; a 3B won't always comply |
| **Repair a missing dot** (`vegancakesco` → `vegancakes.co`)    | the model forgets the dot |
| **De-duplicate, cap at 10**                                    | the model repeats or returns fragments |

That's the *entire* list. There is **no** forced keyword, **no** synthesized name, **no**
category word-list. If nothing about creativity or word-choice is in that table, it's the
model's job, driven by the description.

### What this means in practice (honest limits)
- Names are **varied and on-idea** because we stopped forcing words into them.
- The **key subject word can occasionally drop** on a creative query (e.g. "vegan"). We accept
  this — trying to force it back is exactly the trap that caused keyword-stuffing. It's not
  about 100%; it's about the model understanding the input and responding to it.
- On a **dense or sensitive** query, a 3B can only do so much — it produces reasonable keyword
  names, not a human brand strategist's metaphor. The safety filter keeps stigmatizing words out
  (occasionally leaving fewer than 10 results, which we surface rather than pad).

---

## Part 2 — The flow: how the query and rules reach the SLM

```
  your query
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. QueryLens  (hybrid.analyze)                                        │
│    on-device spaCy model + regex constraints  →  a structured BRIEF   │
│    { intent, concept[], style[], given_name[], qualifiers[],          │
│      require_token[], exclude_token[], constraints{tld,length,…} }    │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. build_user_msg(brief)  →  a SHORT plain description, by intent     │
│      • exact / plain name → "already named X; give close variations"  │
│      • name + concepts    → "keep <name>; pair with one related word" │
│      • creative           → "Describe: <idea>. Feel <style>. For <who>"│
│    Hard constraints (tld / length / hyphen / digits / require /       │
│    exclude) are appended as short one-liners. No "use word X" rules.  │
└─────────────────────────────────────────────────────────────────────┘
      │  + system prompt = the generic "how to name well" quality bar
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. call_slm(system, user)   →  Qwen2.5-3B-Instruct                    │
│    temperature 0.3, top_p 0.8  (measured: cleanest, least drift).     │
│    Returns ~10 raw candidate names as text.                           │
└─────────────────────────────────────────────────────────────────────┘
      │  raw text (often imperfect JSON)
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. parse_domains(text)   →  tolerant extraction                       │
│    tries JSON first; if the model broke the JSON, falls back to a     │
│    regex that pulls domain-like tokens so nothing is lost.            │
└─────────────────────────────────────────────────────────────────────┘
      │  list of candidate strings
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. enforce(candidates, brief)  →  HARD, CHECKABLE CONSTRAINTS ONLY    │
│    a. exact-match first ....... first 1–2 = the exact name (.com/.co) │
│    b. dot repair ............. "vegancakesco" → "vegancakes.co"       │
│    c. TLD ................... force required TLD / fix junk /          │
│                               keep creative extensions sensible       │
│    d. filters .............. length, no-hyphen, no-digits,            │
│                               require/exclude, STIGMA safety, junk     │
│    e. de-dup + cap 10                                                  │
│    f. .com floor ........... ensure ≥6 end in .com (unless a TLD is    │
│                               required or .com is excluded)            │
│    (no forced keywords · no synthesized names · no category lists)    │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
   final domains  →  shown in the demo with per-stage timing
```

### Which file does which step
| Step | Function | File |
|------|----------|------|
| 1. Query → brief | `hybrid.analyze` | `intent_model/hybrid.py` (+ `intent_layer/constraints.py`) |
| 2. Brief → description | `build_user_msg` | `slm_generate.py` |
| 3. Call the model | `call_slm` | `slm_generate.py` |
| 4. Parse the reply | `parse_domains` | `slm_generate.py` |
| 5. Enforce constraints | `enforce` | `slm_generate.py` |
| Demo wiring + timing | `generate` | `demo/engine.py`, `demo/serve.py` |

### The dividing line (why it's split this way)
- **The model owns meaning** — reading the query (QueryLens) and inventing brandable names (SLM).
  Both are things models are genuinely good at.
- **Code owns only what must be exactly true** — the name you already have, the TLD, the length,
  the banned words, and one safety guard. These are cheap to check and expensive to get wrong,
  so they are never left to a 3B model — and nothing *beyond* them is dictated by code.

That separation — a clean, simple input plus a thin layer of hard guarantees — is what lets a
small, cheap model produce reliable, on-brief results without being micro-managed into
keyword soup.
