"""serve.py — zero-dependency web demo (Python stdlib http.server).

Serves a single page where you type a query and watch the full pipeline:
  intent classification  ->  concept/style extraction  ->  generation plan
  ->  (mock SLM candidates)  ->  filter / TLD-quota / re-rank  ->  final domains.

Run:   python3 solution/demo/serve.py     then open http://localhost:8000
No FastAPI/uvicorn needed. Uses the real intent_layer + real post-processor.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine  # noqa: E402

PORT = int(os.getenv("DEMO_PORT", "8000"))

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>QueryLens — Domain Search Query Understanding</title>
<link rel=preconnect href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Mulish:wght@400;600;700;800&display=swap" rel=stylesheet>
<style>
 :root{--bg:#ffffff;--panel:#ffffff;--line:#e4e7e9;--ink:#1c1e21;--mut:#6b7176;
   --teal:#00a3a8;--teal-ink:#006e72;--tealbg:#e6f7f7;--dark:#111820;
   --violet:#6d4bd8;--violetbg:#efeafc;--amber:#8a6d00;--amberbg:#fdf3d6;--redbg:#fde8e6;--red:#b42318}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
   font-family:'Mulish',-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
   font-size:16px;line-height:1.55;-webkit-font-smoothing:antialiased}
 .wrap{max-width:860px;margin:0 auto;padding:36px 20px 60px}
 .brand{display:flex;align-items:center;gap:10px;margin:0 0 2px}
 .dot{width:14px;height:14px;border-radius:50%;background:var(--teal);display:inline-block}
 h1{font-size:30px;font-weight:800;letter-spacing:-.02em;margin:0}
 .sub{color:var(--mut);margin:6px 0 26px;font-size:15px;max-width:640px}
 form{display:flex;gap:10px;margin-bottom:12px}
 input{flex:1;min-width:0;padding:14px 16px;border-radius:12px;border:1.5px solid var(--line);
   background:#fff;color:var(--ink);font-size:16px;font-family:inherit;outline:none}
 input:focus{border-color:var(--teal);box-shadow:0 0 0 3px var(--tealbg)}
 button{padding:14px 22px;border:0;border-radius:12px;background:var(--dark);
   color:#fff;font-weight:700;font-size:16px;font-family:inherit;cursor:pointer;white-space:nowrap}
 button:hover{background:#000}
 .chips{margin:4px 0 24px;display:flex;gap:8px;flex-wrap:wrap}
 .chip{font-size:13px;color:var(--teal-ink);background:var(--tealbg);border:1px solid #cfeeee;
   border-radius:999px;padding:6px 12px;cursor:pointer}
 .chip:hover{background:#d6f2f2}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;
   padding:20px 22px;margin:14px 0;box-shadow:0 1px 2px rgba(17,24,32,.04)}
 .card h3{margin:0 0 14px;font-size:12px;font-weight:700;letter-spacing:.08em;
   text-transform:uppercase;color:var(--mut)}
 .qtext{font-size:17px;font-weight:600;color:var(--ink);
   overflow-wrap:anywhere;word-break:break-word;white-space:normal}
 .badge{display:inline-block;padding:5px 14px;border-radius:999px;font-weight:800;
   font-size:14px;letter-spacing:.02em}
 .b-creative{background:var(--violetbg);color:var(--violet)}
 .b-exact{background:var(--tealbg);color:var(--teal-ink)}
 .b-ambiguous{background:var(--amberbg);color:var(--amber)}
 .row{display:grid;grid-template-columns:130px 1fr;gap:10px;align-items:start;
   padding:10px 0;border-top:1px solid #f0f2f3}
 .row:first-of-type{border-top:0}
 .rk{color:var(--mut);font-size:14px;font-weight:600;padding-top:3px}
 .rv{display:flex;gap:8px;flex-wrap:wrap;min-width:0}
 .tag{background:#f4f6f7;border:1px solid var(--line);border-radius:8px;padding:4px 11px;
   font-size:14px;overflow-wrap:anywhere;word-break:break-word;max-width:100%}
 .tag.concept{background:var(--tealbg);border-color:#cfeeee;color:var(--teal-ink)}
 .tag.style{background:var(--violetbg);border-color:#e0d8f7;color:var(--violet)}
 .tag.excl{background:var(--redbg);border-color:#f6d2ce;color:var(--red)}
 .tag.req{background:#e9f6ec;border-color:#c9e8d0;color:#1a7f37}
 .tag.cons{background:var(--amberbg);border-color:#f0e2b0;color:var(--amber)}
 .muted{color:var(--mut)}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:2px}
 @media(max-width:640px){.grid{grid-template-columns:1fr} .row{grid-template-columns:1fr}}
 .big{font-size:36px;font-weight:800;color:var(--dark)}
 .unit{font-size:14px;color:var(--mut);font-weight:600;margin-left:6px}
 .flow{color:var(--mut);font-size:14px} .flow b{color:var(--ink);font-weight:700}
</style></head><body><div class=wrap>
 <div class=brand><span class=dot></span><h1>QueryLens → Domain Generation</h1></div>
 <p class=sub>Type a domain-search request. QueryLens extracts a structured brief (intent, concept,
   style, given name, qualifiers, must-include / must-avoid, constraints); the SLM then generates
   10 tailored domains from it. Shows the brief, the domains, and total generation time.</p>
 <form onsubmit="go(event)">
   <input id=q placeholder="e.g. a fun short name for my vegan bakery for kids, avoid the word green" autofocus>
   <button>Analyze</button>
 </form>
 <div class=chips id=examples></div>
 <div id=out></div>
</div>
<script>
const EX=["suggest a dominant domain for the gym","The Lounge Coffee Co",
 "a fun catchy name for my vegan bakery for kids","a data engineering brand but avoid the word ai",
 "a name for my clinic in austin, max 12 characters, .shop only",
 "portfolio site for barbers and nail artists"];
const ec=document.getElementById('examples');
EX.forEach(e=>{const s=document.createElement('span');s.className='chip';s.textContent=e;
  s.onclick=()=>{document.getElementById('q').value=e;go();};ec.appendChild(s);});
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function go(ev){ if(ev)ev.preventDefault();
 const q=document.getElementById('q').value.trim(); if(!q)return;
 const out=document.getElementById('out'); out.innerHTML='<div class=card>generating…</div>';
 const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({query:q})}); const d=await r.json();
 if(d.fatal){out.innerHTML='<div class=card>error: '+esc(d.fatal)+'</div>';return;} render(d);
}
function tags(arr,cls){return (arr&&arr.length)?arr.map(x=>`<span class="tag ${cls||''}">${esc(x)}</span>`).join(''):'<span class=muted>—</span>';}
function row(k,v){return `<div class=row><div class=rk>${k}</div><div class=rv>${v}</div></div>`;}
function render(d){
 const T=d.timing||{}, R=d.resources||{};
 const cons=Object.entries(d.constraints||{})
   .filter(([k])=>k!=='require_token'&&k!=='exclude_token')
   .map(([k,v])=>`<span class="tag cons">${esc(k)}: ${esc(Array.isArray(v)?v.join(', '):String(v))}</span>`)
   .join('')||'<span class=muted>—</span>';
 const doms=(d.domains&&d.domains.length)
   ? '<ol style="margin:0;padding-left:24px">'+d.domains.map(x=>`<li style="padding:4px 0;font-size:16px;overflow-wrap:anywhere">${esc(x)}</li>`).join('')+'</ol>'
   : `<div class=muted>no domains — ${d.error?('generation failed: '+esc(d.error)):'—'}</div>`;
 document.getElementById('out').innerHTML=`
 <div class=card><h3>Query</h3><div class=qtext>${esc(d.query)}</div></div>
 <div class=card><h3>Generated domains</h3>${doms}</div>
 <div class=card>
   <h3>Structured brief (QueryLens)</h3>
   ${row('Intent',`<span class="badge b-${d.intent}">${d.intent.toUpperCase()}</span>`)}
   ${row('Concept',tags(d.concept,'concept'))}
   ${row('Style',tags(d.style,'style'))}
   ${row('Given name',tags(d.given_name))}
   ${row('Qualifiers',tags(d.qualifiers))}
   ${row('Must include',tags(d.require_token,'req'))}
   ${row('Must avoid',tags(d.exclude_token,'excl'))}
   ${row('Constraints',cons)}
 </div>
 <div class=grid>
   <div class=card><h3>Total time — final generation</h3>
     <div class=big>${T.total_ms}<span class=unit>ms total</span></div>
     <div class=flow style="margin-top:6px">QueryLens <b>${T.querylens_ms}</b> ms · SLM + enforce <b>${T.generation_ms}</b> ms</div>
   </div>
   <div class=card><h3>Compute footprint</h3>
     <div class=flow><b>${esc(R.device)}</b> · ${R.cpu_cores} cores</div>
     <div class=flow>peak memory <b>${R.process_peak_memory_mb} MB</b></div>
     <div class=flow style="margin-top:6px">${esc(R.components)}</div>
   </div>
 </div>`;
}
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/generate":
            return self._send(404, "not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            res = engine.generate((data.get("query") or "").strip() or "a coffee shop")
            self._send(200, json.dumps(res))
        except Exception as e:
            self._send(500, json.dumps({"fatal": str(e)}))


def main():
    print(f"Warming up models…")
    engine.analyze("warmup coffee shop", repeats=1)   # load spaCy/model once before serving
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    print(f"\n  ▶  Demo ready:  http://localhost:{PORT}\n     Ctrl-C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
