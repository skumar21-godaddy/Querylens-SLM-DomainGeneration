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

SYSTEM = """You are a brand-name generator. Do exactly what the user asks. Reply with ONLY \
this JSON and nothing else: {"domains": ["name1.com", ... 10 names]}"""


_STOP = {"the", "a", "an", "for", "and", "co", "my", "our", "of", "to", "in", "with"}


def build_user_msg(brief: dict) -> str:
    """Translate the structured brief into SIMPLE, concrete English — a small model
    follows plain instructions far better than a JSON schema + abstract rules.
    EXACT intent -> variations of the actual name; CREATIVE -> brandable names."""
    concept = brief.get("concept") or []
    given = brief.get("given_name") or []
    c = brief.get("constraints", {})

    # ---- EXACT: the business already has a name; give close variations of IT ----
    name_src = given or (concept if brief.get("intent") == "exact" else [])
    if name_src:
        name = name_src[0]
        L = [f'A business is already named "{name}". Give 10 domain names that are close '
             f'variations of THIS name.',
             "Vary only by: dropping small words (the, co, and), reordering the words, "
             "joining them, or adding a short ending like hq/online/shop. Keep the main words.",
             "Do NOT invent new or unrelated names."]
        if c.get("tld"):
            L.append(f'End every name with "{c["tld"]}".')
        else:
            L.append("At least 6 must end in .com; a couple can be .co.")
        L.append("Lowercase, no spaces, a dot before the ending (name.com). Reply with JSON only.")
        return "\n".join(L)

    # ---- CREATIVE: brandable names from the concept ----
    subj = ", ".join(concept) or "a business"
    L = [f"Make 10 short, catchy domain names for {subj}."]
    # concept-word retention (the fix for 'vegan' getting dropped)
    words = [w for c in concept for w in c.split() if w.lower() not in _STOP and len(w) > 2]
    if words:
        kw = words[0] + (f'" or "{words[1]}' if len(words) > 1 else "")
        L.append(f'Put the word "{kw}" in most of the names so people know what it is.')
    style = brief.get("style") or []
    if style:
        L.append(f"Make them feel {', '.join(style)}, but do NOT put those words in the names.")
    quals = brief.get("qualifiers") or []
    if quals:
        L.append(f"They should appeal to {', '.join(quals)}.")
    if brief.get("require_token"):
        L.append(f'Every name must contain the word "{brief["require_token"][0]}".')
    if brief.get("exclude_token"):
        L.append(f'Never use the word(s): {", ".join(brief["exclude_token"])}.')
    c = brief.get("constraints", {})
    if c.get("tld"):
        L.append(f'End every name with "{c["tld"]}".')
    else:
        L.append("At least 6 must end in .com; a few can be .co or .shop.")
    if c.get("length_max"):
        L.append(f'Keep each name under {c["length_max"]} letters.')
    if c.get("no_hyphen"):
        L.append("No hyphens.")
    if c.get("no_digits"):
        L.append("No numbers.")
    L.append("Use whole real words, lowercase, no spaces. Write each as name.com "
             "(always a dot before the ending). Reply with JSON only.")
    return "\n".join(L)


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
    """Deterministically enforce the QueryLens constraints the SLM is unreliable on:
    valid single TLD, tld override, length, hyphen/digit, require/exclude, dedup,
    the >=4 .com floor, and the guaranteed first-2 exact match. Nothing else."""
    import re
    c = brief.get("constraints", {})
    tld = c.get("tld")
    lmax, lmin = c.get("length_max"), c.get("length_min")
    excl = [t.lower() for t in brief.get("exclude_token", [])]
    req = [t.lower() for t in brief.get("require_token", [])]
    no_hy, no_dig = c.get("no_hyphen"), c.get("no_digits")

    # GUARANTEE the first 2 are exact-match when the user has a name (exact intent, or
    # creative + given_name). We know the exact string, so build it deterministically.
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
        if "." not in d:
            continue
        label, _, ext = d.partition(".")
        ext = ext.split(".")[-1]                      # collapse any stacked TLD -> last piece
        if tld:
            ext = tld.lstrip(".")                     # force required TLD
        elif ext not in _KNOWN_TLDS:                  # SLM merged junk (funcom) -> real TLD
            ext = "com"
        if not re.fullmatch(r"[a-z0-9-]+", label) or len(label) < 3:
            continue                                   # drop junk fragments like "co"
        if no_hy and "-" in label: continue
        if no_dig and any(ch.isdigit() for ch in label): continue
        if lmax and len(label) > lmax: continue
        if lmin and len(label) < lmin: continue
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
