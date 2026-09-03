"""engine.py — demo backend: QueryLens brief -> SLM domain generation.

Runs QueryLens (intent + spans + constraints), feeds the brief to the SLM, enforces
constraints on the output, and returns the brief + generated domains + timings.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION = os.path.dirname(HERE)                 # repo root (holds slm_generate.py)
sys.path.insert(0, SOLUTION)
sys.path.insert(0, os.path.join(SOLUTION, "intent_model"))
sys.path.insert(0, os.path.join(SOLUTION, "intent_layer"))

import hybrid                # QueryLens analyzer
import slm_generate as SG    # QueryLens brief -> SLM domain generation


def _rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def resources():
    try:
        import spacy
        sv = spacy.__version__
    except Exception:
        sv = "n/a"
    return {"device": "CPU only (no GPU)", "cpu_cores": os.cpu_count(),
            "python": platform.python_version(),
            "process_peak_memory_mb": round(_rss_mb(), 1),
            "components": "on-device NLP model + rule-based constraints (no per-request LLM)"}


def analyze(query, repeats=20):
    a = hybrid.analyze(query)
    fn = getattr(hybrid.analyze, "__wrapped__", hybrid.analyze)
    samples = []
    for _ in range(max(repeats, 1)):
        t = time.perf_counter(); hybrid.analyze(query); samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    lat = {"median_ms": round(samples[len(samples) // 2], 2),
           "mean_ms": round(sum(samples) / len(samples), 2),
           "p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 2)}
    return {"query": query,
            "intent": a["intent"],
            "concept": a["concept"], "style": a["style"], "given_name": a["given_name"],
            "qualifiers": a["qualifiers"],
            "require_token": a.get("require_token", []), "exclude_token": a.get("exclude_token", []),
            "constraints": a["constraints"],
            "latency": lat, "resources": resources()}


def generate(query):
    """QueryLens brief -> SLM -> enforced domains, with per-stage + total timing."""
    t0 = time.perf_counter()
    brief = hybrid.analyze(query)                       # QueryLens
    t_ql = (time.perf_counter() - t0) * 1000
    domains, err = [], None
    t1 = time.perf_counter()
    try:
        raw, _ = SG.call_slm(SG.SYSTEM, SG.build_user_msg(brief))
        m = raw[raw.index("{"): raw.rindex("}") + 1]
        domains = SG.enforce(json.loads(m).get("domains", []), brief)   # constraint enforcement
    except Exception as e:
        err = str(e)
    t_gen = (time.perf_counter() - t1) * 1000
    total = (time.perf_counter() - t0) * 1000
    return {"query": query,
            "intent": brief["intent"], "concept": brief["concept"], "style": brief["style"],
            "given_name": brief["given_name"], "qualifiers": brief["qualifiers"],
            "require_token": brief.get("require_token", []),
            "exclude_token": brief.get("exclude_token", []),
            "constraints": brief["constraints"],
            "domains": domains, "error": err,
            "timing": {"querylens_ms": round(t_ql, 1), "generation_ms": round(t_gen, 1),
                       "total_ms": round(total, 1)},
            "resources": resources()}


def _fmt(r):
    print(f"\nQUERY: {r['query']!r}")
    print(f"  intent      : {r['intent']}  (via {r['intent_source']})")
    print(f"  concept     : {r['concept']}")
    print(f"  style       : {r['style']}")
    print(f"  given_name  : {r['given_name']}")
    print(f"  qualifiers  : {r['qualifiers']}")
    print(f"  constraints : {r['constraints']}")
    print(f"  latency     : {r['latency']['median_ms']} ms median | compute: "
          f"{r['resources']['device']}, {r['resources']['process_peak_memory_mb']} MB")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "suggest a fun name for my vegan bakery for kids"
    _fmt(analyze(q))
