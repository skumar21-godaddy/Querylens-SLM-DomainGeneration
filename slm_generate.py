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

# SLM chat-completions endpoint — set to your own via env SLM_URL (default is a placeholder).
SLM_URL = os.environ.get("SLM_URL", "https://your-slm-endpoint.example.com/v1/chat/completions")
MODEL_VARIANT = os.environ.get("SLM_VARIANT", "Qwen2-5-3B-Instruct")

SYSTEM = """You turn a domain BRIEF into brandable domain names. Output ONLY strict JSON: {"domains":[10 strings]}. Nothing else.

Brief: concept, style, given_name, qualifiers, require_token, exclude_token, constraints{tld,length_max,length_min,no_hyphen,no_digits,position}.

Rules:
1. Exactly 10 distinct names. Each = label + ONE real TLD (name.com). lowercase, no spaces, one dot only.
2. Build every name AROUND the concept so the brand is obvious. Keep the main concept word (or a close real synonym) READABLE in most names — use WHOLE real words; do NOT drop letters or merge words into unpronounceable fragments (a buyer must recognise the concept). Pair the concept word with ONE more real, evocative word or a short suffix (-ly, -hub, -co, -ery). Keep names short and real-sounding. style = tone; qualifier = light flavor.
3. Safe and clean only — no offensive, adult, or disturbing meaning.
4. TLDs: at least 4 end in .com; for the rest use .co, .shop, or .store (or another TLD that clearly fits the concept). Do NOT use .io/.ai/.tech/.app unless the concept is software or tech. TLDs are suffixes only — never put these words inside the name.
5. If constraints.tld is set: every domain uses that TLD (ignore rule 4).
6. require_token in every label; exclude_token in none; no_hyphen; no_digits; length within length_max/length_min; position start/end.

JSON only, 10 items."""


def build_user_msg(brief: dict) -> str:
    keep = {k: brief[k] for k in
            ("intent", "concept", "style", "given_name", "qualifiers",
             "require_token", "exclude_token", "constraints") if k in brief}
    return ("Structured brief:\n" + json.dumps(keep, ensure_ascii=False) +
            "\n\nGenerate exactly 10 domain recommendations. JSON only.")


def call_slm(system: str, user: str, max_tokens=500, temperature=0.5, top_p=0.85):
    # lower temp + top_p keep the small model on coherent tokens (whole words, less
    # gibberish); kept mild so the 10 names still vary. Tune via env if needed.
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
        if not re.fullmatch(r"[a-z0-9-]+", label) or not label:
            continue
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
    # .com floor: ensure >=4 .com unless a specific TLD is required or .com is excluded
    if not tld and "com" not in excl:
        n_com = sum(1 for d in out if d.endswith(".com"))
        i = 0
        while n_com < 4 and i < len(out):
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
    try:
        m = out[out.index("{"): out.rindex("}") + 1]
        raw = json.loads(m).get("domains", [])
    except Exception:
        print(f"SLM ({dt:.2f}s) — unparseable:", out.strip()[:400]); return
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
