"""constraints.py — separate structural INSTRUCTIONS from the concept.

A query like "…real estate company which shall have builder word at end" carries
a constraint (the name must END with the token "builder"), not a concept. Left in
the concept, "word builder" wrongly seeds the SLM. This module pulls constraints
into a structured dict AND returns the concept-only text (instruction clause
stripped) so keyword extraction sees just the subject.

Constraints captured (best-effort, regex — missing one just omits a hint):
  required_token : a token the name must contain ("builder", "fresh", "peak")
  position       : "start" | "end"  (where the required token/prefix-suffix goes)
  no_hyphen / no_digits : character rules
  length_range   : [min, max] characters
  tld            : an explicitly requested TLD (".com")
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

_STOP = {"the", "a", "an", "at", "end", "start", "word", "it", "name", "names",
         "domain", "my", "for", "and", "with", "of", "to", "that", "which",
         "shall", "should", "must", "have", "be", "in", "is", "are"}

# marks the start of a genuine INSTRUCTION/CONSTRAINT clause — text before it is the
# concept. Deliberately NARROW: a bare relative pronoun ("...company THAT manages
# operational impacts") is a DESCRIPTIVE clause (part of the concept), NOT a
# constraint, so we only strip on a relative pronoun + MODAL ("which SHALL have"),
# or an explicit constraint phrase (ends with / at end / no hyphens / only .com).
_CLAUSE = re.compile(
    r"\b(?:"
    r"which (?:shall|must|should|has to|have to|needs? to|will|can|only)|"
    r"that (?:shall|must|should|has to|have to|needs? to|only)|"
    r"(?:it )?(?:shall|must|should) (?:have|be|start|end|contain|include|only)|"
    r"make sure|"
    r"end(?:s|ing)? with|start(?:s|ing)? with|begin(?:s|ning)? with|"
    r"at (?:the )?end|at (?:the )?start|at (?:the )?beginning|"
    r"no (?:hyphens?|dashes?|numbers?|digits?)|"
    r"only \.?com|\.com only|"
    r"with the word|the word\b"
    r")", re.I)

# Required-token ("must contain X") phrasings, ordered by REAL frequency mined from
# both files' data (the word X:37, X in it:27, starts/ends with X:23/9, include X:7,
# X as a word:1). Each targets a general PATTERN, not one query. Capture up to two
# words for the token; a stopword guard drops "that/this/it".
_REQ_PATTERNS = [
    r"\bthe words?\s+([a-z0-9]+(?:\s+[a-z0-9]+)?)",                      # "the word ceda"
    r"\b(?:ends?|ending)\s+(?:in|with)\s+(?:the word\s+)?([a-z0-9]+(?:\s+[a-z0-9]+)?)",
    r"\b(?:starts?|starting|begins?|beginning)\s+with\s+(?:the word\s+)?([a-z0-9]+(?:\s+[a-z0-9]+)?)",
    r"\b(?:include[s]?|contain[s]?|using|use)\s+(?:the word\s+)?([a-z0-9]+(?:\s+[a-z0-9]+)?)",
    r"\bwith\s+(?:the word\s+)?([a-z0-9]+(?:\s+[a-z0-9]+)?)\s+in it\b",   # "with X in it"
    r"\b([a-z0-9]+(?:\s+[a-z0-9]+)?)\s+in it\b",                          # "X in it"
    r"\b(?:have|has|having)\s+([a-z0-9]+)\s+as a word\b",                 # "have x as a word"
    r"\b([a-z0-9]+)\s+as a word\b",                                       # "x as a word"
    r"\bhave\s+([a-z0-9]+)\s+word\b",                                     # "have builder word"
]
# words that are never a meaningful required token
_REQ_STOP = {"that", "this", "it", "the", "a", "an", "word", "words", "name", "domain",
             "all", "them", "some", "any", "my", "your", "in", "with"}
# negation cue shortly BEFORE a "contain/include X" phrase flips it from a REQUIRED
# token to an EXCLUDED one ("shall not contain ai", "without ai", "no ai", "avoid ai").
_NEG = re.compile(r"(?:\bnot\b|n['’]?t\b|\bno\b|\bwithout\b|\bnever\b|\bavoid\b|"
                  r"\bexcludes?\b|\bexcluding\b|\bdon['’]?t\b)", re.I)


_POS_END = re.compile(
    r"\bat (?:the )?end\b|\bends?\s+with\b|\bending\s+(?:in|with)\b|\bsuffix\b|\bat last\b", re.I)
_POS_START = re.compile(
    r"\bat (?:the )?(?:start|beginning|front)\b|\bstarts?\s+with\b|"
    r"\bstarting\s+with\b|\bbegins?\s+with\b|\bprefix\b", re.I)
_NO_HYPHEN = re.compile(r"\bno\s+(?:hyphens?|dashes?)\b", re.I)
_NO_DIGIT = re.compile(r"\bno\s+(?:numbers?|digits?)\b", re.I)
_LENGTH = re.compile(r"\b(\d+)\s*(?:-|to|–|and)\s*(\d+)\s*(?:chars?|characters|letters)\b", re.I)
# single-bound length: "max 15 characters", "under 12 chars", "at most 10 letters",
# "no more than 8 chars", "up to 15 characters", "15 characters or less".
_LEN_MAX = re.compile(
    r"(?:max(?:imum)?|under|below|at most|no more than|up to|less than|<=?)\s*(\d+)\s*"
    r"(?:chars?|characters|letters)\b"
    r"|\b(\d+)\s*(?:chars?|characters|letters)\s*(?:or less|or fewer|max(?:imum)?)\b", re.I)
_LEN_MIN = re.compile(
    r"(?:min(?:imum)?|at least|no fewer than|no less than|over|more than|>=?)\s*(\d+)\s*"
    r"(?:chars?|characters|letters)\b", re.I)
# PRICE / BUDGET ceiling: a money cue ($, dollar, price, value, budget, cost) near a
# max-comparator + number → price_max. Requires a money cue so "under 5 letters" or
# "below the fold" never match. Captures "$ value below 800", "dollar value below
# 800", "under $800", "budget of 500", "less than $1,000".
_PRICE = re.compile(
    r"(?:\$|dollars?|price|value|budget|cost)[^.\d]{0,15}?"
    r"(?:below|under|less\s+than|up\s+to|max(?:imum)?|<|of|around|about)\s*"
    r"\$?\s*([\d,]+)", re.I)
_PRICE_ALT = re.compile(  # "under $800" / "below $1,000" (currency symbol carries the cue)
    r"(?:below|under|less\s+than|up\s+to|max(?:imum)?|<)\s*\$\s*([\d,]+)", re.I)
# TLD extraction. A leading-dot token is treated as a requested extension ONLY if it
# is a REAL TLD — validated against a comprehensive ASCII/English TLD corpus
# (tld_corpus.txt, ~530 entries). This gives coverage (any real extension: .eco, .law,
# .shop, .travelersinsurance …) AND precision (random dotted words are NOT treated as TLDs).
def _load_tld_corpus():
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tld_corpus.txt")
    try:
        return frozenset(l.strip().lower() for l in open(p) if l.strip())
    except Exception:                                   # fallback if the corpus file is absent
        return frozenset({"com", "net", "org", "io", "co", "ai", "app", "dev", "shop",
                          "store", "online", "site", "xyz", "info", "biz", "tech", "eco"})


_KNOWN_TLDS = _load_tld_corpus()
# a leading-dot extension (2–24 chars to cover long TLDs) + optional request lead-in/
# trailer. The lead-in ("ending in / ends with") is captured so it is stripped from the
# concept and does NOT also fire the position rule. Validity is decided by _KNOWN_TLDS.
_TLD_REQ = re.compile(
    r"(?:(?:maybe\s+)?(?:ending|ends?|end)\s+(?:in|with)\s+|(?:only\s+)?(?:in\s+)?)?"
    r"(?<![a-z0-9])\.([a-z]{2,24})\b"
    r"(?:\s*(?:domains?|ideas?|names?|extensions?|only|please))?", re.I)
# a genuine relative-clause constraint ("which shall have … at end") — remove the
# whole clause, since it is an instruction about the name, not the concept.
_MODAL_CLAUSE = re.compile(
    r"\b(?:which|that)\s+(?:shall|must|should|has to|have to|needs? to|will|can)\b.*", re.I)


def extract(query: str) -> Tuple[Dict, str]:
    """Return (constraints_dict, concept_text).

    Constraints are REMOVED as spans wherever they occur (not truncated) so the
    concept survives whether it comes before OR after the constraints. Example:
    "Available .com only 4-12 chars, no hyphens. App generates explorable scenes"
    → concept text keeps "App generates explorable scenes", constraints capture
    the rest.
    """
    q = query or ""
    c: Dict = {}
    spans = []  # (start, end) spans to blank out of the concept text

    def take(regex, store=None):
        m = regex.search(q)
        if m:
            spans.append(m.span())
            if store is not None:
                c[store] = True
        return m

    # required token (+ its phrase span). Multi-word capture, stopword-guarded;
    # trim trailing stopwords from a 2-word capture ("real estate" ok, "attention in" → "attention").
    for pat in _REQ_PATTERNS:
        m = re.search(pat, q, re.I)
        if not m:
            continue
        tok = m.group(1).lower().strip()
        words = [w for w in tok.split() if w not in _REQ_STOP]
        if not words:
            continue
        # negation shortly before the phrase → EXCLUDE, not require
        # ("shall not contain ai" / "without ai" / "no ai")
        preceding = q[max(0, m.start() - 20):m.start()]
        key = "exclude_token" if _NEG.search(preceding) else "required_token"
        c[key] = " ".join(words)
        spans.append(m.span())
        break
    # TLD FIRST (before position): a leading-dot alpha token IS a requested
    # extension — the dot is the signal, so we accept ANY extension (.eco, .vet,
    # .law …), not only a hardcoded whitelist. Capture the optional lead-in
    # ("ending in / ends with") so it is removed from the concept AND so the
    # position rule below does not misread ".eco" as an SLD-suffix constraint.
    tld_span = None
    for m in _TLD_REQ.finditer(q):                # first dotted token that is a REAL TLD
        ext = m.group(1).lower()
        if ext in _KNOWN_TLDS:
            c["tld"] = "." + ext
            tld_span = m.span()
            spans.append(tld_span)
            break
    # position — but skip a match that is really the TLD lead-in ("ending in .eco")
    def _overlaps_tld(span):
        return tld_span is not None and span[0] < tld_span[1] and tld_span[0] < span[1]
    pe = _POS_END.search(q)
    ps = _POS_START.search(q)
    if pe and not _overlaps_tld(pe.span()):
        c["position"] = "end"; spans.append(pe.span())
    elif ps and not _overlaps_tld(ps.span()):
        c["position"] = "start"; spans.append(ps.span())
    # char rules
    take(_NO_HYPHEN, "no_hyphen")
    take(_NO_DIGIT, "no_digits")
    m = _LENGTH.search(q)
    if m:
        c["length_range"] = [int(m.group(1)), int(m.group(2))]; spans.append(m.span())
    else:                                            # single-bound max/min length
        # ONLY with an explicit direction word (max/under/at least/…). A BARE
        # "15 characters" is ambiguous — "characters" can mean a story's cast, and
        # the direction (exact/max/min) is unknown — so we do NOT guess it.
        mx = _LEN_MAX.search(q)
        mn = _LEN_MIN.search(q)
        if mx:
            c["length_max"] = int(mx.group(1) or mx.group(2)); spans.append(mx.span())
        if mn:
            c["length_min"] = int(mn.group(1)); spans.append(mn.span())
    # price/budget ceiling
    pm = _PRICE.search(q) or _PRICE_ALT.search(q)
    if pm:
        c["price_max"] = int(pm.group(1).replace(",", "")); spans.append(pm.span())
    # whole modal relative-clause ("… which shall have builder word at end")
    m = _MODAL_CLAUSE.search(q)
    if m:
        spans.append(m.span())

    # blank out all constraint spans, leaving the rest (concept) intact
    chars = list(q)
    for s, e in spans:
        for i in range(s, min(e, len(chars))):
            chars[i] = " "
    concept_text = re.sub(r"\s+", " ", "".join(chars)).strip(" ,.-")
    return c, (concept_text or q.strip())


if __name__ == "__main__":
    for q in ["help me with domain name for my real estate company which shall have builder word at end",
              "suggest a name for my bakery that ends with fresh",
              "coffee shop name, no hyphens, .com only, 4-12 chars",
              "a name for my gym starting with peak",
              "fun name for my vegan bakery"]:
        print(q[:55], "->", extract(q))
