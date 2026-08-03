"""E3 local review UI for the uncertainty queue.

The server binds to localhost only and uses no third-party web framework.
Labels are written atomically after every decision.

    python -X utf8 scripts/p44_dagger_review.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>E3 DAgger review</title>
<style>
:root { color-scheme: dark; --bg:#111318; --panel:#191d25; --line:#303744;
  --text:#e8ebf0; --muted:#9ba5b4; --accent:#7ca6ff; --good:#78c69b;
  --warn:#e1b866; --bad:#df7b83; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font:14px/1.45 system-ui,Segoe UI,sans-serif; }
header { position:sticky; top:0; z-index:5; display:flex; gap:16px;
  align-items:center; padding:12px 20px; background:var(--panel);
  border-bottom:1px solid var(--line); }
h1 { margin:0; font-size:18px; } h2,h3 { margin:0 0 8px; }
button { border:1px solid var(--line); background:#222834; color:var(--text);
  border-radius:6px; padding:8px 12px; cursor:pointer; }
button:hover { border-color:var(--accent); }
button.primary { background:var(--accent); color:#0c1424; border-color:var(--accent);
  font-weight:650; }
button.good { border-color:var(--good); } button.warn { border-color:var(--warn); }
button.bad { border-color:var(--bad); }
.progress { color:var(--muted); margin-left:auto; }
main { max-width:1250px; margin:0 auto; padding:18px; display:grid;
  grid-template-columns:1fr 1.35fr; gap:16px; }
.panel { background:var(--panel); border:1px solid var(--line);
  border-radius:8px; padding:14px; }
.meta { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:12px; }
.pill { border:1px solid var(--line); border-radius:999px; padding:3px 8px;
  color:var(--muted); }
.players { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.side { border:1px solid var(--line); border-radius:7px; padding:10px; }
.pokemon { margin:5px 0; padding:7px; background:#202631; border-radius:5px; }
.muted { color:var(--muted); } .small { font-size:12px; }
.options { display:grid; gap:8px; }
.option { width:100%; text-align:left; padding:12px; background:#1e242f; }
.option.selected { border-color:var(--accent); background:#253148; }
.option .idx { display:inline-grid; place-items:center; width:24px; height:24px;
  border-radius:50%; background:#303847; margin-right:7px; }
.secret { color:var(--warn); margin-top:6px; }
.actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
textarea { width:100%; min-height:70px; margin-top:12px; padding:9px;
  color:var(--text); background:#10141b; border:1px solid var(--line);
  border-radius:6px; }
.status { min-height:22px; margin-top:8px; color:var(--muted); }
.raw { white-space:pre-wrap; max-height:200px; overflow:auto;
  color:var(--muted); font-size:12px; }
@media(max-width:850px) { main { grid-template-columns:1fr; }
  .players { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>E3 uncertainty review</h1>
  <button id="prev">Previous</button><button id="next">Next</button>
  <label><input id="reveal" type="checkbox"> reveal clone scores</label>
  <span class="progress" id="progress"></span>
</header>
<main>
  <section class="panel">
    <div class="meta" id="meta"></div>
    <div class="players" id="board"></div>
    <details style="margin-top:12px"><summary>Selection metadata</summary>
      <pre class="raw" id="selectMeta"></pre></details>
  </section>
  <section class="panel">
    <h2>Choose the action you would take</h2>
    <p class="muted" id="constraint"></p>
    <div class="options" id="options"></div>
    <textarea id="note" placeholder="Optional rationale or ambiguity note"></textarea>
    <div class="actions">
      <button class="primary" id="saveHigh">Save high-confidence label</button>
      <button class="warn" id="saveLow">Save low-confidence label</button>
      <button class="bad" id="skip">Skip</button>
    </div>
    <div class="status" id="status"></div>
  </section>
</main>
<script>
let items=[], reviews={}, pos=0, picked=new Set();
const $=id=>document.getElementById(id);
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function poke(p) {
  if(!p) return '<span class="muted">empty</span>';
  const e=(p.energies||[]).join(", ")||"none";
  return `<div class="pokemon"><b>${esc(p.name||"unknown")}</b>
    <span class="muted"> HP ${esc(p.hp)}/${esc(p.max_hp)} · energy ${esc(e)}</span></div>`;
}
function side(title,p) {
  const bench=(p.bench||[]).map(poke).join("")||'<span class="muted">empty</span>';
  const hand=p.hand ? p.hand.map(esc).join(", ") : `${p.hand_count} hidden cards`;
  return `<div class="side"><h3>${title}</h3><div class="small muted">Active</div>${poke(p.active)}
    <div class="small muted">Bench</div>${bench}
    <p><b>Hand:</b> ${hand||"empty"}</p>
    <p><b>Deck:</b> ${p.deck_count} · <b>prizes:</b> ${p.prizes_remaining}</p>
    <details><summary>Discard (${(p.discard||[]).length})</summary>${esc((p.discard||[]).join(", ")||"empty")}</details></div>`;
}
function currentReview() { return reviews[items[pos].id]||null; }
function render() {
  const x=items[pos], r=currentReview();
  picked=new Set(r&&r.status==="labeled" ? r.action : []);
  $("progress").textContent=`${pos+1}/${items.length} · ${Object.values(reviews).filter(x=>x.status==="labeled"&&x.confidence==="high").length} high confidence`;
  $("meta").innerHTML=[
    `turn ${x.board.turn}`,`action ${x.board.turn_action_count}`,
    `type ${x.select_type}`,`context ${x.context}`,
    `margin ${x.boundary_margin.toFixed(4)}`,`entropy ${x.normalized_entropy.toFixed(3)}`,
    r ? `${r.status}${r.confidence?" · "+r.confidence:""}` : "unreviewed"
  ].map(v=>`<span class="pill">${esc(v)}</span>`).join("");
  $("board").innerHTML=side("You",x.board.you)+side("Opponent",x.board.opponent);
  $("constraint").textContent=`Select ${x.min_count} to ${x.max_count} option(s). Outcome and logged action are hidden.`;
  $("selectMeta").textContent=JSON.stringify(x.select,null,2);
  $("note").value=r?.note||"";
  renderOptions();
  $("status").textContent="";
}
function renderOptions() {
  const x=items[pos], reveal=$("reveal").checked;
  $("options").innerHTML=x.options.map(o=>{
    const secret=reveal ? `<div class="secret small">clone ${o.clone_selected?"selected":"did not select"} · logit ${o.score.toFixed(4)} · p ${o.probability.toFixed(3)}</div>`:"";
    return `<button class="option ${picked.has(o.index)?"selected":""}" data-i="${o.index}">
      <span class="idx">${o.index}</span><b>${esc(o.label)}</b>
      <div class="muted small">${esc(JSON.stringify(o.raw))}</div>${secret}</button>`;
  }).join("");
  document.querySelectorAll(".option").forEach(b=>b.onclick=()=>{
    const i=Number(b.dataset.i);
    if(picked.has(i)) picked.delete(i); else picked.add(i);
    renderOptions();
  });
}
async function save(status,confidence="") {
  const x=items[pos], action=[...picked].sort((a,b)=>a-b);
  if(status==="labeled"&&(action.length<x.min_count||action.length>x.max_count)) {
    $("status").textContent=`Choose between ${x.min_count} and ${x.max_count} options.`; return;
  }
  const body={id:x.id,status,confidence,action,note:$("note").value};
  const res=await fetch("/api/label",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const data=await res.json();
  if(!res.ok){ $("status").textContent=data.error||"save failed"; return; }
  reviews=data.reviews; $("status").textContent="Saved.";
  const next=items.findIndex((it,i)=>i>pos&&!reviews[it.id]);
  if(next>=0) pos=next; else if(pos<items.length-1) pos++;
  render();
}
async function init() {
  const data=await (await fetch("/api/data")).json();
  items=data.items; reviews=data.reviews;
  const first=items.findIndex(x=>!reviews[x.id]); pos=first>=0?first:0; render();
}
$("prev").onclick=()=>{ if(pos>0){pos--;render();} };
$("next").onclick=()=>{ if(pos<items.length-1){pos++;render();} };
$("reveal").onchange=renderOptions;
$("saveHigh").onclick=()=>save("labeled","high");
$("saveLow").onclick=()=>save("labeled","low");
$("skip").onclick=()=>save("skipped","");
init().catch(e=>$("status").textContent=e);
</script>
</body></html>
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_queue(path: Path) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    ids = [x.get("id") for x in items]
    if not items or len(set(ids)) != len(ids) or any(not x for x in ids):
        raise SystemExit("queue must contain unique, non-empty item ids")
    return items


def load_reviews(path: Path, queue_sha: str) -> dict[str, Any]:
    if not path.exists():
        return {"experiment": "E3", "queue_sha256": queue_sha, "reviews": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("queue_sha256") != queue_sha:
        raise SystemExit(
            f"{path} belongs to a different queue; move it aside explicitly")
    if not isinstance(data.get("reviews"), dict):
        raise SystemExit(f"{path} has no reviews mapping")
    return data


def atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


def public_item(item: dict[str, Any]) -> dict[str, Any]:
    # The observation is retained in the queue for corpus export but is not
    # needed by the browser. Outcomes are absent from both representations.
    return {k: v for k, v in item.items() if k != "observation"}


def make_handler(items: list[dict[str, Any]], review_path: Path,
                 store: dict[str, Any]):
    by_id = {x["id"]: x for x in items}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "E3Review/1.0"

        def send_bytes(self, body: bytes, content_type: str,
                       status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value: Any,
                      status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_bytes(
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8", status)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self.send_bytes(HTML.encode("utf-8"),
                                "text/html; charset=utf-8")
            elif path == "/api/data":
                with lock:
                    reviews = dict(store["reviews"])
                self.send_json({
                    "items": [public_item(x) for x in items],
                    "reviews": reviews,
                })
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/label":
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > 64_000:
                    raise ValueError("invalid request size")
                value = json.loads(self.rfile.read(length))
                item_id = value.get("id")
                if item_id not in by_id:
                    raise ValueError("unknown item id")
                status = value.get("status")
                if status not in ("labeled", "skipped"):
                    raise ValueError("status must be labeled or skipped")
                confidence = value.get("confidence") or ""
                if status == "labeled" and confidence not in ("high", "low"):
                    raise ValueError("labeled decisions need confidence")
                action = value.get("action") or []
                if not isinstance(action, list) or any(
                        not isinstance(x, int) for x in action):
                    raise ValueError("action must be an integer list")
                action = sorted(set(action))
                item = by_id[item_id]
                if status == "labeled":
                    if any(x < 0 or x >= item["n_options"] for x in action):
                        raise ValueError("action index out of range")
                    if not item["min_count"] <= len(action) <= item["max_count"]:
                        raise ValueError("action count violates select bounds")
                note = str(value.get("note") or "")[:2000]
                record = {
                    "status": status,
                    "confidence": confidence,
                    "action": action if status == "labeled" else [],
                    "note": note,
                }
                with lock:
                    store["reviews"][item_id] = record
                    atomic_write(review_path, store)
                    reviews = dict(store["reviews"])
                self.send_json({"ok": True, "reviews": reviews})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, fmt: str, *args: Any) -> None:
            if args and str(args[1]) != "200":
                super().log_message(fmt, *args)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", default="out/e3/review_queue.jsonl")
    ap.add_argument("--reviews", default="out/e3/reviews.json")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    queue_path = ROOT / args.queue
    review_path = ROOT / args.reviews
    if not queue_path.exists():
        raise SystemExit(
            f"{queue_path} is missing; run scripts/p43_dagger_queue.py first")
    items = read_queue(queue_path)
    store = load_reviews(review_path, digest(queue_path))
    handler = make_handler(items, review_path, store)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    high = sum(r.get("status") == "labeled"
               and r.get("confidence") == "high"
               for r in store["reviews"].values())
    print(f"E3_REVIEW_READY {len(items)} items; {high} high-confidence labels")
    print(f"  {url}")
    print(f"  labels -> {review_path}")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nreview server stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
