"""slm_generate.py — TEST integration: QueryLens brief -> SLM domain generation.

Runs QueryLens (hybrid.analyze) on a query, feeds the structured brief to the Qwen SLM
with a generation-tuned system prompt, and prints 10 tailor-made domain recommendations.
Not part of the standalone QueryLens repo — this is an end-to-end experiment.
"""
from __future__ import annotations

import json, os, ssl, sys, time, urllib.request

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl.create_default_context()
_SSL_INSECURE = ssl._create_unverified_context()   # internal dev endpoint fallback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "intent_model"))
sys.path.insert(0, os.path.join(HERE, "intent_layer"))
import hybrid
from constraints import _KNOWN_TLDS               # reuse QueryLens's 531-TLD corpus

SLM_URL = os.environ.get("SLM_URL", "https://your-slm-endpoint.example.com/v1/chat/completions")
MODEL_VARIANT = os.environ.get("SLM_VARIANT", "Qwen2-5-3B-Instruct")

# The system prompt carries the QUALITY BAR (once, generically), so the per-query message
# can stay a short, plain description. The two are designed to work together: system = how to
# name well; user = what to name. We do NOT dictate which words must appear — the model reads
# the description and decides, which is what keeps output varied instead of keyword-stuffed.
SYSTEM = """You are an expert brand-name generator. You receive a short description of a \
business or idea and return domain names for it.

Make 10 names that:
- sound like real, memorable brands — blend, shorten, or coin words. Do NOT just glue the \
description's words together, and do NOT repeat the same word in most names.
- are short and easy to say (aim for under 15 letters), lowercase, one word where possible.
- capture the ONE core idea. Never copy long phrases, and never list every detail from the \
description.
- skip generic filler ("nonprofit", "company", "organization", "business", "official") and \
never put sensitive or stigmatizing words in a name.

Reply with ONLY this JSON, nothing else: {"domains": ["name1.com", "name2.com", ... 10 total]}"""

# SAFETY guard (not a creativity rule): never surface a name containing a stigmatizing word,
# even if it appears in the query. Kept tiny and chosen to avoid substring false-positives
# (no bare "sex" → "essex/unisex"; no "rapist" → "therapist").
_STIGMA = {"offender", "offenders", "predator", "predators", "svp",
           "felon", "felons", "molest", "pedophile", "incest"}


def build_user_msg(brief: dict) -> str:
    """Turn the brief into a SHORT, plain description of what to name — not a rule list.
    The system prompt already knows how to name well; here we only say WHAT the thing is,
    the way a person would brief a designer. Hard constraints (tld/length/require/exclude)
    are appended as one-liners. We deliberately do NOT tell the model which words to repeat
    — letting it read the description is what keeps the names varied and on-idea."""
    concept = brief.get("concept") or []
    given = brief.get("given_name") or []
    c = brief.get("constraints", {})

    def _tail(lines):                                 # append the hard, checkable constraints
        if c.get("tld"): lines.append(f'Every name must end in {c["tld"]}.')
        else: lines.append("Most should end in .com; a couple can be .co.")
        if brief.get("require_token"):
            lines.append(f'Include the word "{brief["require_token"][0]}" in every name.')
        if brief.get("exclude_token"):
            lines.append(f'Never use: {", ".join(brief["exclude_token"])}.')
        if c.get("length_max"): lines.append(f'Keep each name under {c["length_max"]} letters.')
        if c.get("no_hyphen"): lines.append("No hyphens.")
        if c.get("no_digits"): lines.append("No numbers.")
        return "\n".join(lines)

    # NAME + CONCEPTS: the name is fixed; the concepts say what it's for.
    if given and concept:
        return _tail([f'The name is "{given[0]}". It is used for: {", ".join(concept[:4])}.',
                      f'Keep "{given[0]}" in every name and pair it with one short related word.'])

    # EXACT / plain name: the business is already named; give close variations.
    name_src = given or (concept if brief.get("intent") == "exact" else [])
    if name_src:
        return _tail([f'The business is already named "{name_src[0]}".',
                      "Give close variations of this exact name — shorten it, reorder or join the "
                      "words, or add a short ending. Keep the core words; don't invent new names."])

    # CREATIVE: describe the idea in one line and let the model name it.
    lines = [f"Describe: {', '.join(concept) or 'a small business'}."]
    if brief.get("style"):
        lines.append(f"It should feel {', '.join(brief['style'])}.")
    quals = brief.get("qualifiers") or []
    if quals:
        aud = "; ".join(quals)
        lines.append(f"It is for {aud[:80]}.")           # audience is context, not name material
    return _tail(lines)


def call_slm(system: str, user: str, max_tokens=500, temperature=0.3, top_p=0.8):
    # temp 0.3 / top_p 0.8 measured best: highest concept-word retention + cleanest,
    # most on-topic names. Higher values add randomness and off-concept drift. Env-tunable.
    temperature = float(os.environ.get("SLM_TEMP", temperature))
    top_p = float(os.environ.get("SLM_TOP_P", top_p))
    payload = {"model": "/models", "max_tokens": max_tokens, "temperature": temperature,
               "top_p": top_p,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    req = urllib.request.Request(
        SLM_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Model-Variant": MODEL_VARIANT},
        method="POST")
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60, context=_SSL) as r:
            body = json.loads(r.read())
    except ssl.SSLError:                              # internal dev cert → retry unverified
        with urllib.request.urlopen(req, timeout=60, context=_SSL_INSECURE) as r:
            body = json.loads(r.read())
    dt = time.perf_counter() - t
    return body["choices"][0]["message"]["content"], dt


def parse_domains(text):
    """Extract the domain list from the SLM reply — tolerant of malformed JSON
    (small models sometimes close with ')' or drop brackets). Falls back to a regex
    that pulls domain-like tokens, so a bad delimiter never loses the whole result."""
    import json, re
    try:
        m = text[text.index("{"): text.rindex("}") + 1]
        d = json.loads(m).get("domains")
        if d:
            return d
    except Exception:
        pass
    return re.findall(r'[a-z0-9][a-z0-9-]*\.[a-z]{2,10}\b', text.lower())


def enforce(domains, brief):
    """Enforce ONLY the hard, checkable constraints the SLM is unreliable on — nothing about
    creativity or word choice (that is the model's job, driven by the prompt):
      exact-match first (when the user already has a name), missing-dot repair, a valid single
      TLD (required one / sensible default), length, hyphen/digit, require/exclude, dedup,
      and a >=6 .com floor. No forced keywords, no synthesized names."""
    import re
    c = brief.get("constraints", {})
    tld = c.get("tld")
    lmax, lmin = c.get("length_max"), c.get("length_min")
    excl = [t.lower() for t in brief.get("exclude_token", [])]
    req = [t.lower() for t in brief.get("require_token", [])]
    no_hy, no_dig = c.get("no_hyphen"), c.get("no_digits")

    # GUARANTEE the first 1-2 are the exact name when the user already has one — the one thing
    # we know for certain and the model paraphrases away. (Not a creativity rule.)
    name_src = brief.get("given_name") or (brief.get("concept") if brief.get("intent") == "exact" else [])
    if name_src:
        sld = re.sub(r"[^a-z0-9]", "", name_src[0].lower())
        if sld and (not lmax or len(sld) <= lmax):
            primary = tld.lstrip(".") if tld else "com"
            exact = [f"{sld}.{primary}"] + ([] if tld else [f"{sld}.co"])
            domains = exact + [d for d in domains if str(d).strip().lower().strip(".") not in exact]

    out, seen = [], set()
    for d in domains:
        d = str(d).strip().lower().strip(".")
        if "." not in d:                              # SLM often returns a bare name ("VitaGear")
            d = d + ".com"                            # it's a brand name — add the default TLD
        label, _, ext = d.partition(".")
        ext = ext.split(".")[-1]                      # collapse any stacked TLD -> last piece
        if tld:
            ext = tld.lstrip(".")                     # force required TLD
        elif ext not in _KNOWN_TLDS:                  # SLM merged junk (funcom) -> real TLD
            ext = "com"
        elif not name_src and ext not in ("com", "co", "shop", "store"):
            ext = "co"                                # creative: keep TLDs sensible (no .name/.tech)
        if not re.fullmatch(r"[a-z0-9-]+", label) or len(label) < 3:
            continue                                   # drop junk fragments like "co"
        if no_hy and "-" in label: continue
        if no_dig and any(ch.isdigit() for ch in label): continue
        if lmax and len(label) > lmax: continue
        if lmin and len(label) < lmin: continue
        if any(w in label for w in _STIGMA): continue   # safety: drop stigmatizing names
        if any(t in label for t in excl): continue
        if req and not all(t.replace(" ", "") in label for t in req): continue
        if ext in excl:                              # exclude_token named a TLD
            ext = "com" if "com" not in excl else ext
        dom = f"{label}.{ext}"
        if dom not in seen:
            seen.add(dom); out.append(dom)
    out = out[:10]
    # .com floor: ensure >=6 .com unless a specific TLD is required or .com is excluded
    if not tld and "com" not in excl:
        n_com = sum(1 for d in out if d.endswith(".com"))
        i = 0
        while n_com < 6 and i < len(out):
            if not out[i].endswith(".com"):
                cand = out[i].rsplit(".", 1)[0] + ".com"
                if cand not in seen:
                    seen.discard(out[i]); out[i] = cand; seen.add(cand); n_com += 1
            i += 1
    return out


def recommend(query: str, dry: bool = False):
    brief = hybrid.analyze(query)
    user = build_user_msg(brief)
    print(f"\n{'='*74}\nQUERY: {query}")
    print("QueryLens brief:", json.dumps({k: brief[k] for k in
          ("intent", "concept", "style", "given_name", "qualifiers",
           "require_token", "exclude_token", "constraints")}, ensure_ascii=False))
    if dry:                                          # inspect the exact SLM input, no call
        print("\n--- EXACT SLM REQUEST (system + user messages) ---")
        print("[system]\n" + SYSTEM)
        print("\n[user]\n" + user)
        return
    try:
        out, dt = call_slm(SYSTEM, user)
    except Exception as e:
        print(f"  [SLM call failed: {e}]")
        return
    raw = parse_domains(out)
    final = enforce(raw, brief)
    print(f"SLM ({dt:.2f}s) -> {len(raw)} raw, {len(final)} valid after constraint enforcement:")
    for i, d in enumerate(final, 1):
        print(f"   {i:2d}. {d}")
    if len(final) < 10:
        print(f"   [only {len(final)}/10 passed constraints — SLM under-produced valid names]")


if __name__ == "__main__":
    args = sys.argv[1:]
    dry = "--dry" in args                            # show QueryLens brief + exact SLM input, no SLM call
    qs = [a for a in args if a != "--dry"] or [
        "The Lounge Coffee Co",
        "suggest a fun modern name for my vegan bakery for kids",
        "a coffee brand with the word bean in it, .co only",
        "a data engineering brand but avoid the word ai, max 12 characters",
    ]
    for q in qs:
        recommend(q, dry=dry)
