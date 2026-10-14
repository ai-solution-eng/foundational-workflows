#!/usr/bin/env python3
"""Cross-component test: agent_client.py WRITES, frozen stmem.py REAPS.
Proves the two independently-written halves agree on the stmm/v1 wire format.
Mock mirrors the REAL api_server.py shapes, including the subtlety impl-agent-sdk
found: next_cursor is only present when the request carries a cursor param."""
import importlib.util, json, os, subprocess, sys, threading, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STORE, NEXT, REQS = {}, [0], []
ROOT = "/home/andrew/Code/HPE/MultimodalRAG/stmem"

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode())
        docs = body if isinstance(body, list) else [body]
        ids = []
        for d in docs:
            NEXT[0] += 1; pid = "pt-%04d" % NEXT[0]
            STORE[pid] = {"text": d if isinstance(d, str) else json.dumps(d)}; ids.append(pid)
        self._send(200, {"status": "ok", "stored_ids": ids, "count": len(ids)})
    def do_GET(self):
        REQS.append(self.path)
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        limit = int(q.get("limit", ["50"])[0]); has_cursor = "cursor" in q
        cursor = q.get("cursor", [""])[0]
        off = int(cursor.split("-")[1]) if cursor.startswith("cursor-") else 0
        items = list(STORE.items()); page = items[off:off+limit]
        nxt = "cursor-%d" % (off+limit) if off+limit < len(items) else ""
        # REAL server shape: without a cursor param there is NO next_cursor key.
        if has_cursor:
            self._send(200, {"documents": [{"id": i, "payload": p} for i, p in page], "next_cursor": nxt})
        else:
            self._send(200, {"documents": [{"id": i, "payload": p} for i, p in page], "count": len(items)})
    def do_DELETE(self):
        pid = self.path.rstrip("/").split("/")[-1]
        STORE.pop(pid, None); self._send(200, {"status": "ok", "deleted": pid})

def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["STMM_BASE_URL"] = "http://127.0.0.1:%d" % port
    os.environ["STMM_DATASET"] = "swarm-stmem"

    spec = importlib.util.spec_from_file_location("agent_client", ROOT + "/agent_client.py")
    ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

    fails = []
    def check(name, cond, detail=""):
        print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else "  :: " + str(detail)[:400]))
        if not cond: fails.append(name)

    run_id = ac.new_run_id()
    check("run_id is unique-ish/sortable", run_id.startswith("stmm-") and len(run_id) > 12, run_id)

    id1 = ac.write_fact("observed: pod X is CrashLooping in mm-rag", run_id, "agent-1", step=1)
    id2 = ac.write_fact("inferred: probably OOM, not yet confirmed", run_id, "agent-1", step=2,
                        epistemic="inferred", ttl_hours=0)
    check("client wrote 2 entries", len(STORE) == 2 and id1 and id2, (id1, id2, STORE))

    entries = ac.recall(run_id)
    check("recall sees both", len(entries) == 2, entries)
    check("recall ordered by step", [e.get("step") for e in entries] == [1, 2], entries)

    brief = ac.handoff_brief(run_id)
    btext = brief if isinstance(brief, str) else json.dumps(brief)
    check("brief contains observed fact", "CrashLooping" in btext, btext[:200])
    check("brief flags inferred", "INFERRED" in btext.upper(), btext[:300])

    promos = ac.promote_candidates(run_id)
    ptext = promos if isinstance(promos, str) else json.dumps(promos)
    check("promote excludes inferred", "probably OOM" not in ptext, ptext[:300])
    check("promote includes observed", "CrashLooping" in ptext, ptext[:300])

    # CROSS-COMPONENT: frozen reaper must see what the client wrote
    r = subprocess.run([sys.executable, ROOT + "/stmem.py", "list", "--run", run_id],
                       capture_output=True, text=True, timeout=60)
    check("frozen stmem.py list sees client entries", r.returncode == 0 and "CrashLooping" in r.stdout,
          r.stdout + r.stderr)
    check("reaper list shows 2 lines",
          len([l for l in r.stdout.strip().splitlines() if l.strip()]) == 2, r.stdout)

    # CROSS-COMPONENT: reaper deletes the expired (ttl_hours=0) client entry, keeps the live one
    r = subprocess.run([sys.executable, ROOT + "/stmem.py", "reap"], capture_output=True, text=True, timeout=60)
    check("reap exit0", r.returncode == 0, r.stderr)
    check("reaper deleted client's expired entry", id2 not in STORE, STORE)
    check("reaper kept client's live entry", id1 in STORE, STORE)
    check("reap summary says 1 delete", "to_delete=1" in r.stdout, r.stdout)

    check("client always sends cursor= (real-server shape)", any("cursor=" in p for p in REQS), REQS[:3])

    srv.shutdown()
    print("")
    print("RESULT: " + ("ALL PASS" if not fails else "%d FAILED: %s" % (len(fails), fails)))
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())

