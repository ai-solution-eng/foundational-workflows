#!/usr/bin/env python3
"""Lead's adversarial delta probe: N1 range/overflow + N2 non-JSON, via the SHIPPED CLI."""
import json, os, subprocess, sys, threading, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STORE = {}
MODE = {"nonjson": False, "empty": False}
SCRIPT = "/home/andrew/Code/HPE/MultimodalRAG/stmem/stmem.py"

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        if MODE["nonjson"]:
            b = b"<html>gateway error</html>"
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if MODE["empty"]:
            b = b""
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers(); return
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n: self.rfile.read(n)
        self._send({"status": "ok", "stored_ids": ["x"]})
    def do_GET(self):
        items = list(STORE.items())
        self._send({"documents": [{"id": i, "payload": p} for i, p in items], "next_cursor": ""})
    def do_DELETE(self):
        STORE.pop(self.path.rstrip("/").split("/")[-1], None)
        self._send({"status": "ok"})

def env(port):
    return {"STMM_BASE_URL": "http://127.0.0.1:%d" % port, "STMM_DATASET": "t",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin")}

def run(e, *args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, env=e, timeout=60)

def envelope(created_at, ttl="__missing__", supersedes=None, run_id="R"):
    e = {"stmm": 1, "schema": "stmm/v1", "run_id": run_id, "agent": "a", "step": 1,
         "epistemic": "observed", "created_at": created_at}
    if ttl != "__missing__":
        e["ttl_hours"] = ttl
    if supersedes is not None:
        e["supersedes"] = supersedes
    return json.dumps(e) + "\nfact text"

srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
E = env(srv.server_address[1])
fails = []
def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else "  :: " + str(detail)[:220]))
    if not cond: fails.append(name)

def sweep(docs, ttl_arg="24"):
    STORE.clear()
    for i, t in enumerate(docs):
        STORE["pt-%d" % i] = {"text": t}
    return run(E, "reap", "--ttl-hours", ttl_arg)

HEALTHY_EXPIRED = envelope("2020-01-01T00:00:00Z", 24)   # long expired -> must be reaped
HEALTHY_FRESH = envelope("2999-01-01T00:00:00Z", 24)     # far future -> must be kept

# V2: adversarial boundary values alongside a healthy expired entry
cases = [
    ("ttl=876000 (accepted max)", 876000),
    ("ttl=876001 (boundary+1)", 876001),
    ("ttl=1e8", 100000000),
    ("ttl=1e9", 1000000000),
    ("ttl=inf", float("inf")),
    ("ttl=nan", float("nan")),
    ("ttl=-1", -1),
    ("ttl=True", True),
    ("ttl='24' (string)", "24"),
    ("ttl=None", None),
]
for label, ttl in cases:
    r = sweep([envelope("2020-01-01T00:00:00Z", ttl), HEALTHY_EXPIRED])
    completed = r.returncode == 0
    reaped_healthy = "pt-1" in r.stdout and "(expired)" in r.stdout
    check("V2 sweep completes: " + label, completed, r.stderr[:150])
    check("V2 healthy entry still reaped: " + label, reaped_healthy, r.stdout.strip()[:150])

# V2b: extreme created_at values
for label, ts in [("9999-12-31T23:59:59Z", "9999-12-31T23:59:59Z"), ("0001-01-01T00:00:00Z", "0001-01-01T00:00:00Z")]:
    r = sweep([envelope(ts, 876000), HEALTHY_EXPIRED])
    check("V2 sweep completes: created_at " + label, r.returncode == 0, r.stderr[:150])
    check("V2 healthy entry reaped: created_at " + label, "(expired)" in r.stdout, r.stdout.strip()[:150])

# V3: legitimate expiry semantics preserved
r = sweep([HEALTHY_EXPIRED])
check("V3 25h-old ttl=24 IS deleted", "(expired)" in r.stdout and "to_delete=1" in r.stdout, r.stdout)
r = sweep([envelope("2026-10-11T09:00:00Z", 24)])  # ~1h old at probe time
check("V3 1h-old ttl=24 is KEPT", "to_delete=0" in r.stdout, r.stdout)
r = sweep([envelope("2020-01-01T00:00:00Z")], ttl_arg="24")  # ttl key absent
check("V3 missing ttl falls back to --ttl-hours", "(expired)" in r.stdout, r.stdout)

# V4: N2 non-JSON / empty 2xx
MODE["nonjson"] = True
r = run(E, "reap")
check("V4 reap: non-JSON 2xx -> clean error, no traceback",
      r.returncode != 0 and "Traceback" not in r.stderr, r.stderr[:200])
r = run(E, "ensure-dataset")
check("V4 ensure-dataset: non-JSON 2xx -> clean error, no traceback",
      r.returncode != 0 and "Traceback" not in r.stderr, r.stderr[:200])
MODE["nonjson"] = False
MODE["empty"] = True
r = run(E, "ensure-dataset")
check("V4 empty 2xx body -> {} (no error)", r.returncode == 0, r.stdout + r.stderr)
MODE["empty"] = False

srv.shutdown()
print("")
print("RESULT: " + ("ALL PASS" if not fails else "%d FAILED: %s" % (len(fails), fails[:6])))
sys.exit(1 if fails else 0)

