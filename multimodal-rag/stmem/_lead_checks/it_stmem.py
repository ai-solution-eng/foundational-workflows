#!/usr/bin/env python3
"""Integration harness: run stmem.py CLI against a mock of the real mm-rag API."""
import json, os, subprocess, sys, threading, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STORE = {}
NEXT = [0]
REQS = []

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        REQS.append(("POST", self.path))
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode())
        docs = body if isinstance(body, list) else [body]
        ids = []
        for d in docs:
            NEXT[0] += 1
            pid = "pt-%04d" % NEXT[0]
            STORE[pid] = {"text": d if isinstance(d, str) else json.dumps(d)}
            ids.append(pid)
        self._send(200, {"status": "ok", "stored_ids": ids, "count": len(ids)})
    def do_GET(self):
        REQS.append(("GET", self.path))
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        limit = int(q.get("limit", ["50"])[0])
        cursor = q.get("cursor", [""])[0]
        off = int(cursor.split("-")[1]) if cursor.startswith("cursor-") else 0
        items = list(STORE.items())
        page = items[off:off + limit]
        nxt = "cursor-%d" % (off + limit) if off + limit < len(items) else ""
        self._send(200, {"documents": [{"id": i, "payload": p} for i, p in page], "next_cursor": nxt})
    def do_DELETE(self):
        REQS.append(("DELETE", self.path))
        pid = self.path.rstrip("/").split("/")[-1]
        existed = pid in STORE
        STORE.pop(pid, None)
        self._send(200, {"status": "ok", "deleted": pid} if existed else {"status": "ok"})

SCRIPT = "/home/andrew/Code/HPE/MultimodalRAG/stmem/stmem.py"

def run(*args, env_extra=None):
    env = dict(os.environ); env.update(env_extra or {})
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, env=env, timeout=60)

def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = {"STMM_BASE_URL": "http://127.0.0.1:%d" % port, "STMM_DATASET": "swarm-stmem"}
    fails = []
    def check(name, cond, detail=""):
        print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else "  :: " + str(detail)[:300]))
        if not cond: fails.append(name)

    r = run("write", "--run", "runA", "--agent", "agent-1", "--step", "1", "--text", "observed: pod X is CrashLooping", env_extra=env)
    check("write#1 exit0", r.returncode == 0, r.stderr)
    fresh = json.loads(r.stdout)["stored_ids"][0]

    r = run("write", "--run", "runA", "--agent", "agent-1", "--step", "2", "--ttl-hours", "0", "--text", "inferred: probably OOM", "--epistemic", "inferred", env_extra=env)
    check("write#2 exit0", r.returncode == 0, r.stderr)
    expired = json.loads(r.stdout)["stored_ids"][0]

    r = run("write", "--run", "runA", "--agent", "agent-2", "--step", "3", "--text", "observed: pod X restarted, OOM confirmed", "--supersedes", fresh, env_extra=env)
    check("write#3 exit0", r.returncode == 0, r.stderr)
    check("3 entries stored", len(STORE) == 3, STORE)

    r = run("list", "--run", "runA", env_extra=env)
    check("list exit0", r.returncode == 0, r.stderr)
    check("list shows 3 lines", len([l for l in r.stdout.strip().splitlines() if l.strip()]) == 3, r.stdout)

    before = dict(STORE)
    r = run("reap", "--dry-run", env_extra=env)
    check("reap --dry-run exit0", r.returncode == 0, r.stderr)
    check("dry-run deleted nothing", STORE == before, STORE)
    check("dry-run reports to_delete=2", "to_delete=2" in r.stdout, r.stdout)

    r = run("reap", env_extra=env)
    check("reap exit0", r.returncode == 0, r.stderr)
    check("expired removed", expired not in STORE, STORE)
    check("superseded removed", fresh not in STORE, STORE)
    check("live entry kept", len(STORE) == 1, STORE)

    STORE["manual-1"] = {"text": "a plain long-term memory doc, no envelope at all"}
    STORE["manual-2"] = {"text": json.dumps({"stmm": 1, "run_id": "runA", "agent": "x", "created_at": "NOT-A-DATE", "ttl_hours": 24}) + chr(10) + "broken expiry"}
    r = run("reap", env_extra=env)
    check("reap exit0 (round 2)", r.returncode == 0, r.stderr)
    check("INVARIANT envelope-less doc kept", "manual-1" in STORE, STORE)
    check("INVARIANT unparseable-expiry doc kept", "manual-2" in STORE, STORE)

    check("pagination GETs seen", any("cursor=" in p for m, p in REQS if m == "GET"), REQS[-6:])

    r = run("reap", env_extra={"STMM_BASE_URL": ""})
    check("no base url -> clean error", r.returncode != 0 and "STMM_BASE_URL" in (r.stdout + r.stderr), r.stdout + r.stderr)

    srv.shutdown()
    print("")
    print("RESULT: " + ("ALL PASS" if not fails else "%d FAILED: %s" % (len(fails), fails)))
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())

