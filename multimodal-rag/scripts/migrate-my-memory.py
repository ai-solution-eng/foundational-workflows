#!/usr/bin/env python3
"""Fran's memory migration v3 -- export -> native import (text-replay).

Upgrades a memory dataset to the hybrid (dense + bm25) schema by round-tripping
it through the app's own export/import. The export carries payloads only (no
vectors), so the import re-embeds every document -- which is what rebuilds the
bm25 keyword vectors.

  0. prompts for the dataset password (getpass -- never stored, never logged)
  1. exports the CURRENT dataset -> <dataset>-export-<ts>.tar.gz
     (rollback: the complete, restorable copy -- keep it)
  2. shows what it found (counts, kinds, file-like docs)
  3. after a typed confirmation: POSTs the export to
     /api/admin/datasets/import (multipart: file, overwrite=true, password)
     -- one call; the server replays documents.jsonl through the embedder
     (file-based datasets go through the media recreate flow instead)
  4. polls the import job, then verifies document_count / schema_version /
     has_password

The restored dataset is UNPROTECTED unless the password is passed to the
import (the export strips the password hash by design) -- this script always
passes it.

Aborting before step 3 leaves the dataset untouched.
"""

import datetime
import getpass
import gzip
import io
import json
import os
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.request

API = "https://rag-mcp-server.pcai-se-ai-application.hst.rdlabs.hpecorp.net"
DATASET = "francesco-memory"
# Fleet API key (REST endpoints): prompted or env-provided, never a literal.
# Resolution order: RAG_API_KEY env -> interactive getpass. (The historical
# hardcoded key was revoked after the P0-9 secret-scan gate flagged it in
# git HEAD and the delivery tree; mint a fresh key with scripts/mint_key.py
# or the /access page and pass it via RAG_API_KEY.)
API_KEY = os.environ.get("RAG_API_KEY", "").strip() or getpass.getpass("RAG API key: ").strip()
if not API_KEY:
    print("no API key provided -- aborting, nothing was changed.")
    sys.exit(1)


def build_opener():
    ca = (
        os.environ.get("RAG_CA_BUNDLE")
        or os.environ.get("NODE_EXTRA_CA_CERTS")
        or os.path.expanduser("~/.config/opencode/pcai-aie-root-ca.crt")
    )
    if ca and os.path.exists(ca):
        ctx = ssl.create_default_context(cafile=ca)
        print("TLS: using CA", ca)
    else:
        ctx = ssl.create_default_context()
        print("TLS: system store (set RAG_CA_BUNDLE=<pcai-aie-root-ca.crt> if verification fails)")
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


OPENER = build_opener()

PASSWORD = getpass.getpass("dataset password for " + DATASET + ": ")
AUTH = {"X-RAG-Api-Key": API_KEY, "X-Dataset-Password": PASSWORD}


def api(method, path, timeout=60, data=None, headers=None):
    req = urllib.request.Request(API + path, method=method, data=data, headers=headers or AUTH)
    return OPENER.open(req, timeout=timeout)


# -- step 1: export the LIVE dataset (rollback + the content we restore) -----
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print("step 1/4: exporting live dataset ...")
try:
    body = api("GET", "/api/datasets/" + DATASET + "/export", timeout=180).read()
except urllib.error.HTTPError as e:
    print("  export failed (HTTP " + str(e.code) + "): " + e.read().decode()[:300])
    if e.code in (401, 403):
        print("  password rejected -- nothing was changed.")
    sys.exit(1)
gz = body[:2] == b"\x1f\x8b"
export_file = DATASET + "-export-" + stamp + (".tar.gz" if gz else ".json")
open(export_file, "wb").write(body)  # rollback = exact bytes as served
data = gzip.decompress(body) if gz else body
print("  saved rollback:", export_file, "(" + str(len(body)) + " bytes)")

# -- step 2: parse + census ---------------------------------------------------
docs = None
try:  # tar export: meta.json + documents.jsonl
    tf = tarfile.open(fileobj=io.BytesIO(data))
    member = next((m for m in tf.getmembers() if m.name.endswith("documents.jsonl")), None)
    handle = tf.extractfile(member) if member is not None else None
    if handle is not None:
        docs = [json.loads(line) for line in handle.read().decode("utf-8").splitlines() if line.strip()]
        print("  (tar export; parsed", len(docs), "docs from documents.jsonl)")
except tarfile.ReadError:
    pass
if docs is None:
    try:  # plain json
        d = json.loads(data)
        docs = (
            d
            if isinstance(d, list)
            else next((d[k] for k in ("documents", "memories", "data", "points") if isinstance(d.get(k), list)), None)
        )
    except Exception:
        try:  # jsonl
            docs = [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]
        except Exception:
            docs = None
if not isinstance(docs, list) or not docs:
    print("  could not locate a document list in the export.")
    print("  ABORT -- nothing was changed. Send this output back for a parser fix.")
    sys.exit(1)


def unwrap(d):
    if isinstance(d, dict) and isinstance(d.get("payload"), dict):
        d = d["payload"]
    return {
        "text": d.get("text") or d.get("page_content") or d.get("content") or "",
        "metadata": d.get("metadata") or {},
    }


items = [unwrap(d) for d in docs]
textish = [m for m in items if (m["text"] or "").strip()]
fileish = [m for m in items if not (m["text"] or "").strip()]
print("step 2/4: export holds", len(items), "documents:", len(textish), "text +", len(fileish), "file/media")
kinds: dict[str, int] = {}
for m in items:
    k = str((m["metadata"] or {}).get("memory_kind"))
    kinds[k] = kinds.get(k, 0) + 1
print("  by memory_kind:", dict(sorted(kinds.items())))
if fileish:
    print("  (file/media docs: the import rebuilds them via the media recreate")
    print("   flow -- originals, captions and tiers are regenerated, not lost.)")
print("  rollback export saved:", export_file)

ans = input("  type REBUILD to re-embed everything via native import, anything else aborts: ").strip()
if ans != "REBUILD":
    print("aborted -- nothing was changed.")
    sys.exit(1)

# -- step 3: native import (overwrite + password; re-embeds = bm25 rebuild) ---
print("step 3/4: importing", export_file, "->", DATASET, "(overwrite=true) ...")
boundary = "----franMigration" + stamp + "XXXX"
crlf = b"\r\n"
parts = []
for k, v in (("overwrite", "true"), ("password", PASSWORD)):
    parts.append(
        b"--"
        + boundary.encode()
        + crlf
        + f'Content-Disposition: form-data; name="{k}"'.encode()
        + crlf
        + crlf
        + str(v).encode()
        + crlf
    )
parts.append(
    b"--"
    + boundary.encode()
    + crlf
    + f'Content-Disposition: form-data; name="file"; filename="{export_file}"'.encode()
    + crlf
    + b"Content-Type: application/gzip"
    + crlf
    + crlf
    + body
    + crlf
)
parts.append(b"--" + boundary.encode() + b"--" + crlf)
mp_body = b"".join(parts)
try:
    resp = api(
        "POST",
        "/api/admin/datasets/import",
        timeout=300,
        data=mp_body,
        headers={"X-RAG-Api-Key": API_KEY, "Content-Type": "multipart/form-data; boundary=" + boundary},
    )
    result = json.loads(resp.read())
except urllib.error.HTTPError as e:
    print("  import failed (HTTP " + str(e.code) + "): " + e.read().decode()[:300])
    print("  The dataset may be in a transitional state -- the rollback export")
    print("  (" + export_file + ") is the restorable copy; re-run the import with it.")
    sys.exit(1)
print("  import accepted:", {k: result.get(k) for k in ("job_id", "status", "dataset", "mode")})
job_id = result.get("job_id", "")
if not job_id:
    print("  no job_id in response -- check the server; rollback:", export_file)
    sys.exit(1)

# -- step 4: poll + verify ----------------------------------------------------
TERMINAL_OK = ("complete", "completed", "done", "success", "finished")
TERMINAL_BAD = ("failed", "error", "cancelled", "canceled", "aborted")


def _env_float(name, default):
    """Read a numeric knob defensively: a malformed value falls back to *default*."""
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def poll_import_status(job_id, fetch_status, total_files=0):
    """Poll an import job to a TERMINAL status and return its last status dict.

    The previous version polled 60 x 10 s and then declared failure while the
    server-side import kept running: a full memory-dataset re-embed legitimately
    exceeds ten minutes, and the operator was told the migration failed when it
    was still succeeding.  Poll to a terminal status instead, with an overall cap
    sized from the job's own total_files (a re-embed takes seconds per file, so
    30 s/file plus ten minutes of slack) and overridable with
    RAG_IMPORT_MAX_WAIT (seconds) / RAG_IMPORT_POLL_INTERVAL (seconds).

    Returns None only when the cap is reached -- the server-side import may still
    be running, which the caller reports as "check manually", never as failure.
    """
    cap_seconds = _env_float("RAG_IMPORT_MAX_WAIT", 600.0 + 30.0 * max(0, int(total_files or 0)))
    interval = _env_float("RAG_IMPORT_POLL_INTERVAL", 10.0)
    deadline = time.monotonic() + cap_seconds
    last = None
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        time.sleep(interval)
        try:
            last = fetch_status()
        except Exception as e:  # transient poll failure: keep polling
            print("  poll", attempt, "failed:", str(e)[:120])
            continue
        status = str(last.get("status", ""))
        print("  poll", attempt, ":", status, "| processed:", last.get("processed_files"), "/", last.get("total_files"))
        if last.get("error"):
            return last
        if status.lower() in TERMINAL_OK or status.lower() in TERMINAL_BAD:
            return last
    return None


print("step 4/4: polling job", job_id, "...")
final = poll_import_status(
    job_id,
    lambda: json.loads(api("GET", "/api/datasets/" + DATASET + "/upload-status/" + job_id, timeout=30).read()),
    total_files=len(items),
)
if final is None:
    print(
        "  polling window exhausted (RAG_IMPORT_MAX_WAIT) -- the import may still be"
        " running; check GET /api/datasets/" + DATASET + "/upload-status/" + job_id + " manually; rollback:",
        export_file,
    )
    sys.exit(1)
if final.get("error"):
    print("  IMPORT JOB ERROR:", str(final.get("error"))[:300])
    print("  rollback:", export_file)
    sys.exit(1)
if str(final.get("status", "")).lower() in TERMINAL_BAD:
    print("  IMPORT JOB ENDED:", str(final.get("status", "")) or "(no status)", "-- rollback:", export_file)
    sys.exit(1)
res = final.get("result") or {}
print("  job result:", {k: res.get(k) for k in ("status", "dataset", "mode", "documents")})

print("  verifying dataset info ...")
info = json.loads(api("GET", "/api/datasets/" + DATASET, timeout=30).read()).get("dataset", {})
print(
    "  dataset:",
    {k: info.get(k) for k in ("name", "document_count", "schema_version", "has_password", "embedder_model")},
)
print("  expected document_count ~", len(items))
print()
print("DONE -", DATASET, "rebuilt via text-replay: dense re-embedded, bm25 rebuilt.")
print("Final check is yours: run a memory search for a term only in an OLD memory.")
