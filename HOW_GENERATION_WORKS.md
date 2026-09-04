# How Domain Generation Works

This file explains, in two parts, exactly what happens when you type a query and get
10 domains back:

- **Part 1 — Plain English.** What the SLM is asked to do, what *we* guarantee with our
  own rules, and who is responsible for what. No code.
- **Part 2 — The flow.** How the query becomes a brief, how the brief becomes a prompt,
  what gets sent to the SLM, and how the SLM's answer is cleaned up before you see it.

The whole pipeline is: **your query → QueryLens brief → English prompt → SLM → our rules → 10 domains.**

---

## Part 1 — Plain English

### The idea in one line
A small language model (SLM) is *creative but unreliable*. So we let it brainstorm names,
but we never trust it to follow rules. Anything that **must** be true about the final list,
we enforce ourselves in plain code afterwards.

### Who does what

**QueryLens (our on-device model) reads your query** and turns it into a short structured
"brief": the intent (exact / creative / ambiguous), the concept ("vegan bakery"), the style
("fun", "modern"), a given name ("bean"), who it's for, words that must be included or avoided,
and hard constraints (a required `.com`, a length limit, "no hyphens", etc.).

**We write the SLM a simple English instruction** built from that brief. Small models follow
plain sentences far better than a big list of rules, so the prompt is short and concrete, and
it changes depending on intent:

- **Exact** ("The Lounge Coffee Co", "lawyers association of USA LLC") — the business already
  has a name. We ask only for *close variations* of that exact name (drop small words, reorder,
  join, add a short ending like `hq`/`online`). We explicitly tell it **not** to invent new names.
- **Name + concepts** ("our server is named *bean*, we use it for minecraft, dns, jellyfin") —
  here the name is the anchor and the concepts describe what it does. We ask for names that
  **start with the name and add one short word about those concepts** (`beanserver`, `beandns`).
- **Creative** ("a fun name for my vegan bakery for kids") — we ask for short, catchy names,
  tell it to keep the important concept word ("vegan") in most of them, and to make them *feel*
  the style ("fun") **without literally using the style words** in the name.

**The SLM generates ~10 candidate names.** It's good at the creative spark ("fruityveggies",
"cheerfulcarrots", "brewhaven") but it regularly breaks rules: it drops the key word, invents
weird prefixes, forgets the dot before the TLD, uses odd extensions, or ignores a length limit.

**Then our rules take over and guarantee the things that actually matter.** This is the part
we do *not* leave to the model:

| What we force (deterministic rules)                         | Why the SLM can't be trusted with it |
|-------------------------------------------------------------|--------------------------------------|
| **Exact-match first**: if you have a name, the first 1–2 results are the exact name itself (`.com` and `.co`) | The model paraphrases the name away |
| **Crux word stays in ≥4 names** (creative / name+concepts)  | The model keeps dropping "vegan" / ignoring "bean" |
| **Required TLD / at least 6 `.com`**                        | The model scatters random extensions |
| **Sensible extensions** (no `.name`/`.tech` on a bakery)    | The model picks irrelevant TLDs |
| **Length limits, no-hyphen, no-digits**                     | The model ignores numeric limits |
| **Must-include / must-avoid words** (`require`/`exclude`)   | The model uses a word you banned |
| **Repair a missing dot** (`vegancakesco` → `vegancakes.co`) | The model forgets the dot |
| **Drop junk, de-duplicate, cap at 10**                      | The model repeats or returns fragments |

### What "forced" means (and what it is *not*)
The crux guarantee is the subtle one, so to be precise: we do **not** blindly staple the key
word onto the model's names (that produced ugly pileups like `veganplantbasedbake`). Instead we
**synthesize clean names from your own brief** — the key word plus a *relevant* word that was
already in your query (`beanserver`, `beandns`, `veganbakery`, `vegankids`) — and use those to
replace only the weakest model suggestions. If the brief has no other usable word, we fall back
to a small pool of neutral endings (`hub`, `spot`, `goods`). We never overwrite the exact-match
names, and we skip this entirely for *exact* intent (there, "the name + variations" is the point).

Because these names are built by us, generation stays correct **even when the SLM misbehaves** —
if it returns garbage, you still get `beanserver`/`beandns`/`beanhub`.

### The short version
- **SLM = creativity.** It proposes names.
- **Rules = guarantees.** They make sure the name you asked for, the word that matters, the TLD,
  the length, and the banned words are all correct — every time, regardless of the model.

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
│ 2. build_user_msg(brief)   →  a short ENGLISH prompt, chosen by intent│
│      • exact / plain name  → "give close variations of THIS name"     │
│      • name + concepts     → "start with <name>, add a word about …"  │
│      • creative            → "catchy names, keep <concept word>,      │
│                               feel <style> but don't spell it out"    │
│    Hard constraints (tld / length / hyphen / digits / require /       │
│    exclude) are appended as plain sentences.                          │
└─────────────────────────────────────────────────────────────────────┘
      │  system prompt = "reply with ONLY this JSON: {domains:[…10…]}"
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. call_slm(system, user)    →  Qwen2.5-3B-Instruct                   │
│    temperature 0.3, top_p 0.8  (measured: best word-retention, least  │
│    drift). Returns ~10 raw candidate names as text.                   │
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
│ 5. enforce(candidates, brief)   →  THE DETERMINISTIC RULES            │
│    a. exact-match prepend .... first 1–2 = the exact name (.com/.co)  │
│    b. dot repair ............... "vegancakesco" → "vegancakes.co"     │
│    c. TLD .................... force required TLD / fix junk /         │
│                                keep creative extensions sensible      │
│    d. filters ............... length, no-hyphen, no-digits,           │
│                                require/exclude words, drop fragments   │
│    e. de-dup + cap 10                                                  │
│    f. .com floor ............ ensure ≥6 end in .com (unless a TLD is   │
│                                required or .com is excluded)           │
│    g. CRUX floor ............ ensure the key word appears in ≥4 names, │
│                                using clean brief-driven combos         │
│                                (beanserver / veganbakery); skipped for │
│                                exact intent; never touches (a).        │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
   final 10 domains  →  shown in the demo with per-stage timing
```

### Which file does which step
| Step | Function | File |
|------|----------|------|
| 1. Query → brief | `hybrid.analyze` | `intent_model/hybrid.py` (+ `intent_layer/constraints.py`) |
| 2. Brief → English prompt | `build_user_msg` | `slm_generate.py` |
| 3. Call the model | `call_slm` | `slm_generate.py` |
| 4. Parse the reply | `parse_domains` | `slm_generate.py` |
| 5. Enforce the rules | `enforce` | `slm_generate.py` |
| Demo wiring + timing | `generate` | `demo/engine.py`, `demo/serve.py` |

### The dividing line (why it's split this way)
- **The model owns meaning** — reading the query (QueryLens) and inventing brandable names (SLM).
  Both are things models are genuinely good at.
- **Code owns anything that must be exactly true** — the name you already have, the word that
  must survive, the TLD, the length, the banned words. These are cheap to check and expensive to
  get wrong, so they are never left to a 3B model.

That separation is what lets a small, cheap model produce reliable, on-brief results.
