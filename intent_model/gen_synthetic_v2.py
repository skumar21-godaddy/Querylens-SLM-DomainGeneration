"""gen_synthetic_v2.py — generate ~800 synthetic domain-search queries (Opus) that
COVER the rare/hard patterns real data under-supplies, then label them with the same
v2 teacher (label_v2) so labels are schema-consistent.

Queries are 20-35 words, realistic, varied industries, occasional typos/run-ons.
Writes labels_v2_synth.jsonl. Resumable on the labeling step.
"""
from __future__ import annotations

import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label_v2 as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "labels_v2_synth.jsonl")
RAW = os.path.join(HERE, "synth_queries.txt")

# pattern buckets -> (target count, style instruction)
BUCKETS = [
    (120, "the user names an AUDIENCE or LOCATION they serve (for kids, for small "
          "businesses in canada, targeting busy nurses) plus a described business"),
    (80,  "the user requires the NAME to CONTAIN a specific word (must include the word "
          "'cloud', with 'fresh' in it, use the word summit)"),
    (130, "the user says the name must NOT contain certain words, using varied negation "
          "(without the word crypto, avoid anything with ai or tech, not containing "
          "'shop', don't use generic luxury words) — vary how far the negation sits from the word"),
    (100, "aspirational / emotional brand briefs with little concrete product (mountains "
          "are my dream, I want to inspire people, a movement about belonging)"),
    (120, "the user states a name they ALREADY have, via 'named X', 'called X', 'the name "
          "shall be X', 'my company is X', or a domain like example.com, then asks for help"),
    (100, "two or more businesses/services coordinated with 'along with', 'as well as', "
          "'plus', or 'together with'"),
    (60,  "a run-on list of professions/items with FEW or NO commas between them, some misspelled"),
    (90,  "the core concept is the OBJECT of a verb inside a 'that/which/who' clause "
          "(a platform that connects tutors with students, an app which tracks workouts)"),
]

GEN_SYSTEM = """You generate REALISTIC natural-language domain-search queries a real \
GoDaddy user would type. Vary industry, length (20-35 words), tone, and phrasing. \
Include occasional typos, run-ons, and lowercase — like real users. Do NOT number them. \
Output ONE query per line, nothing else."""


def generate(client, n, style):
    msg = client.messages.create(
        model=L.MODEL, max_tokens=4096, system=GEN_SYSTEM,
        messages=[{"role": "user", "content":
                   f"Generate {n} distinct domain-search queries where {style}. "
                   f"Each 20-35 words. One per line."}])
    lines = [re.sub(r"^\s*[-\d.)]+\s*", "", x).strip()
             for x in msg.content[0].text.splitlines() if x.strip()]
    return [x for x in lines if len(x.split()) >= 8]


def main():
    cli = L._client()
    # 1) generate (cache raw queries so we don't regenerate on rerun)
    if os.path.exists(RAW):
        queries = [l.strip() for l in open(RAW) if l.strip()]
    else:
        queries, seen = [], set()
        for n, style in BUCKETS:
            got = []
            for _ in range(3):                      # up to 3 tries to reach target
                if len(got) >= n:
                    break
                for q in generate(cli, n - len(got), style):
                    k = re.sub(r"\s+", " ", q.lower())
                    if k not in seen:
                        seen.add(k); got.append(q)
                time.sleep(0.5)
            print(f"  bucket({n}) -> {len(got)}", flush=True)
            queries += got[:n]
        with open(RAW, "w") as f:
            f.write("\n".join(queries))
    print(f"[synth] generated {len(queries)} queries -> {RAW}", flush=True)

    # 2) label with the v2 teacher (resumable)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                done.add(json.loads(l)["text"])
            except Exception:
                pass
    todo = [q for q in queries if q not in done]
    print(f"[synth] label todo={len(todo)}", flush=True)
    with open(OUT, "a") as f:
        for k in range(0, len(todo), 15):
            for r in L.label_batch(todo[k:k + 15], cli):
                r["synthetic"] = True
                f.write(json.dumps(r) + "\n")
            f.flush()
    print(f"[synth] DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
