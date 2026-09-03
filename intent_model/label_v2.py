"""label_v2.py — OFFLINE teacher labeler, EXPANDED 6-label schema.

Labels intent + tight spans for: CONCEPT, STYLE, GIVEN_NAME, QUALIFIER,
REQUIRE_TOKEN, EXCLUDE_TOKEN. Closed-form constraints (tld/length/price/hyphen)
are handled by regex at serve time, NOT labeled here. Reads CAAS_AUTH_TOKEN env.
Batched + resumable (see bulk driver). This SYSTEM prompt is the critical artifact.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import anthropic

# LLM teacher endpoint + model — set to your own provider via env (offline use only).
BASE_URL = os.environ.get("CAAS_BASE_URL", "https://your-llm-endpoint.example.com")
MODEL = os.environ.get("CAAS_MODEL", "claude-opus-4-8")

SYSTEM = """You label natural-language DOMAIN-SEARCH queries to train a small spaCy \
NER + intent model for GoDaddy. Return STRICT JSON only.

intent — exactly one:
  "exact"     = the query IS a specific brand/business/product NAME or a domain the \
user already has ("The Lounge Coffee Co", "fileclerk.com"). They want that name.
  "creative"  = the user DESCRIBES a business/concept/need and wants name/domain IDEAS.
  "ambiguous" = genuinely between: a given name PLUS a request, OR too vague/short, OR \
a list of candidate names, OR off-topic/support.

spans — TIGHT, minimal, VERBATIM substrings. Extract EVERY applicable span. Labels:

  CONCEPT = the subject/product/theme/deliverable being named. 1-3 words EACH; output \
a SEPARATE CONCEPT for every distinct one (split coordinated lists, incl. those joined \
by "and", commas, "along with", "as well as", "plus"). Include concepts that are the \
OBJECT of a verb inside a "that/which/who" clause ("a site that hosts videos" -> \
"videos"). Keep short industry acronyms (ai, ml, crm, seo, saas). Do NOT put the \
audience/location here (that is QUALIFIER). Do NOT include abstract fluff words \
(excellence, integrity, passion) or naming words (domain/name/brand).

  STYLE = adjective describing the desired NAME itself: fun, modern, short, premium, \
minimalist, edgy, catchy, brandable, invented, futuristic. "invented/made-up words" \
means STYLE=invented, NOT a required token.

  GIVEN_NAME = a specific name the user ALREADY has. The ACTUAL proper name ONLY — for \
"the name shall be Dignity Moves Foundation" the span is "Dignity Moves Foundation" \
(NOT "the name"). Include domains ("forkliftsunltd.com") and suffixes ("Acme Widgets \
LLC"). Keep it SHORT — never a whole descriptive/aspirational clause.

  QUALIFIER = the audience / target-market / location / who-or-where it serves: "kids", \
"canada", "small businesses", "barbers", "for weddings", "in maryland". This is NOT the \
concept. In "supplement for kids" -> CONCEPT "supplement", QUALIFIER "kids".

  REQUIRE_TOKEN = a specific word the NAME MUST contain: "the word cloud", "must include \
fresh", "with X in it". The token itself.

  EXCLUDE_TOKEN = a specific word the name must NOT contain, signalled by negation \
("not contain ai", "without crypto", "avoid watch time chrono", "no X"). The token \
itself. Read the negation from context even when it is several words away.

Do NOT label these (a regex handles them): TLD/extension (.com, .io), length \
("4-12 chars"), price/budget ("under $800"), no-hyphen/no-digit, start/end position.

Rules: every span a verbatim substring; prefer FEWER, SHORTER, meaningful spans; omit \
labels that don't apply. For a vague/aspirational sentence with no concrete subject, \
give a TIGHT CONCEPT (e.g. "mountains") or none — never a runaway GIVEN_NAME.

Output ONLY a JSON array, one object per query, in order:
[{"i":0,"intent":"creative","spans":[{"label":"CONCEPT","text":"vegan bakery"},{"label":"STYLE","text":"fun"},{"label":"QUALIFIER","text":"kids"}]}]"""


def _client():
    return anthropic.Anthropic(base_url=BASE_URL, auth_token=os.environ["CAAS_AUTH_TOKEN"])


def label_batch(queries: List[str], client=None, retries: int = 3) -> List[Dict]:
    client = client or _client()
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(queries))
    user = f"Label these {len(queries)} queries:\n{numbered}"
    last = None
    for attempt in range(retries):
        try:
            msg = client.messages.create(model=MODEL, max_tokens=4096, system=SYSTEM,
                                         messages=[{"role": "user", "content": user}])
            arr = json.loads(re.search(r"\[.*\]", msg.content[0].text, re.S).group(0))
            out = [None] * len(queries)
            for obj in arr:
                i = obj.get("i")
                if isinstance(i, int) and 0 <= i < len(queries):
                    out[i] = {"text": queries[i], "intent": obj.get("intent", "ambiguous"),
                              "spans": obj.get("spans", [])}
            for i, q in enumerate(queries):
                if out[i] is None:
                    out[i] = {"text": q, "intent": "ambiguous", "spans": []}
            return out
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"label_batch failed: {last}")
