#!/usr/bin/env python3
"""stmem.py — short-term (swarm) memory client + reaper for MultimodalRAG.

stdlib only. Env:
  STMM_BASE_URL   e.g. http://rag-mcp-server-api.mm-rag.svc.cluster.local:80  (required)
  STMM_API_KEY    sent as BOTH X-API-Key and X-RAG-Api-Key if set
  STMM_PASSWORD   sent as X-Dataset-Password if set
  STMM_DATASET    default: swarm-stmem
  ST_TTL_HOURS    default: 24

Usage:
  stmem.py write --run R --agent A --step 3 --text "fact..." [--epistemic observed|inferred]
                 [--ttl-hours 24] [--supersedes id1,id2]
  stmem.py list  --run R
  stmem.py export --run R            # JSONL of entries (for the distiller/promoter)
  stmem.py reap [--ttl-hours 24] [--dry-run]
  stmem.py delete --ids id1,id2
  stmem.py ensure-dataset            # idempotent; run before reap in-cluster

Robustness contract (review fixes, 2026-10-11):
  * A malformed entry can NEVER abort a sweep or stop other entries being reaped
    (all per-entry errors are contained; unparseable => KEEP).
  * The reaper always terminates, even if the server returns a repeating cursor.
  * Transport errors on one delete do not abandon the remaining deletes.
  * Any transport/HTTP failure surfaces as ONE stderr line and a non-zero exit,
    never a traceback.
"""
import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

BASE = os.environ.get("STMM_BASE_URL", "").rstrip("/")
DATASET = os.environ.get("STMM_DATASET", "swarm-stmem")
API_KEY = os.environ.get("STMM_API_KEY")
PASSWORD = os.environ.get("STMM_PASSWORD")

_DECODER = json.JSONDecoder()

# A TTL beyond this is treated as malformed rather than trusted (N1): the old
# code only type-checked, so an out-of-range value reached the arithmetic and
# raised an uncaught OverflowError that killed the whole sweep.
_MAX_TTL_HOURS = 876000  # 100 years


def _env_ttl_default() -> int:
    """ST_TTL_HOURS must be a plain integer; a bad value is a clean error."""
    raw = os.environ.get("ST_TTL_HOURS", "24")
    try:
        value = int(raw)
    except ValueError:
        sys.exit(f"stmem: ST_TTL_HOURS must be an integer, got {raw!r}")
    if value < 0:
        sys.exit(f"stmem: ST_TTL_HOURS must be >= 0, got {value}")
    return value


def _req(method, path, body=None, timeout=30):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if API_KEY:
        # Live tree accepts X-API-Key; .baseline_head middleware only reads
        # X-RAG-Api-Key / Authorization: Bearer.  Send both (F8).
        req.add_header("X-API-Key", API_KEY)
        req.add_header("X-RAG-Api-Key", API_KEY)
    if PASSWORD:
        req.add_header("X-Dataset-Password", PASSWORD)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except ValueError as ex:
        # A non-JSON 2xx body (e.g. an HTML gateway page) must surface as a
        # clean transport error, not an uncaught JSONDecodeError (N2).
        raise urllib.error.URLError(f"non-JSON response from {method} {path}: {raw[:120]!r}") from ex


def _parse_envelope(text):
    """Return (envelope_dict_or_None, body_text).

    Tolerant: the envelope is the FIRST JSON object in the text (whitespace
    allowed, pretty-printed multi-line objects accepted).  Anything else is
    treated as 'not an stmm entry' and therefore never deleted.
    """
    if not isinstance(text, str):
        return None, text
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return None, text
    try:
        env, end = _DECODER.raw_decode(stripped)
    except ValueError:
        return None, text
    if not isinstance(env, dict) or env.get("stmm") != 1:
        return None, text
    return env, stripped[end:].lstrip("\r\n")


def _find_text(payload):
    """Find the envelope-bearing string in a stored payload.

    Payload layout varies (str, {"page_content": ...}, {"text": ...}).  We
    prefer a value that actually PARSES as an envelope rather than the first
    value merely containing the token (F4: a decoy must not shadow the real
    entry).
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        candidates = [v for v in payload.values() if isinstance(v, str)]
        for v in candidates:
            if _parse_envelope(v)[0] is not None:
                return v
        for v in candidates:
            if '"stmm"' in v:
                return v
    return None


def iter_documents():
    """Yield (id, payload) for every document, terminating even on a bad server."""
    cursor = ""
    seen = set()
    while True:
        qs = f"/api/datasets/{DATASET}/documents?limit=500&cursor={urllib.parse.quote(cursor)}"
        resp = _req("GET", qs)
        for d in resp.get("documents", []):
            yield d.get("id"), d.get("payload")
        cursor = resp.get("next_cursor")
        if not cursor:
            return
        if cursor in seen:
            print(f"WARNING: repeating next_cursor {cursor!r}; stopping pagination", file=sys.stderr)
            return
        seen.add(cursor)


def _parse_created_at(value):
    """Parse the envelope timestamp; tolerate ISO-8601 variants (F3)."""
    if not isinstance(value, str) or not value:
        return None
    v = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(v, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None
    # No zone info => ambiguous (date-only, "2026-10-10T11:00:00") => refuse:
    # fail-closed beats a deletion that lands hours early or late.
    return dt if dt.tzinfo is not None else None


def _ttl_hours(value, fallback):
    """ttl_hours must be a finite number in range; anything else is invalid (F1/N1)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if not math.isfinite(v) or v < 0 or v > _MAX_TTL_HOURS:
        return None
    return v


def _supersedes_of(env):
    """supersedes must be a list/tuple of ids; ignore malformed values (F1)."""
    raw = env.get("supersedes")
    if isinstance(raw, (list, tuple)):
        return [s for s in raw if isinstance(s, str) and s]
    return []


def cmd_ensure(a):
    """Idempotently ensure the scratch dataset exists (safe to run every sweep)."""
    try:
        resp = _req("POST", "/api/datasets", {
            "name": DATASET,
            "description": "ST-MEM swarm short-term memory (ephemeral; default 24h delete)",
        })
        print(f"created dataset {DATASET}: {resp.get('status', 'ok')}")
        return 0
    except urllib.error.HTTPError as ex:
        detail = ""
        try:
            detail = ex.read().decode()[:200]
        except Exception:
            pass
        # The API returns 409 (FileExistsError) for an existing dataset. A 400 is
        # only treated as success when the body explicitly says it exists — the
        # broad substring "exist" would also match "does not exist" (reviewer N2).
        if ex.code == 409 or (ex.code == 400 and "exists" in detail.lower()):
            print(f"dataset {DATASET} already exists")
            return 0
        print(f"FAILED to ensure dataset {DATASET}: HTTP {ex.code} {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as ex:
        print(f"FAILED to ensure dataset {DATASET}: {ex}", file=sys.stderr)
        return 1


def cmd_write(a):
    env = {
        "stmm": 1, "schema": "stmm/v1", "run_id": a.run, "agent": a.agent,
        "step": a.step, "epistemic": a.epistemic,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ttl_hours": a.ttl_hours,
        "supersedes": [s for s in a.supersedes.split(",") if s],
    }
    text = json.dumps(env) + "\n" + a.text
    resp = _req("POST", f"/api/datasets/{DATASET}/documents", [text])
    print(json.dumps({"stored_ids": resp.get("stored_ids"), "run_id": a.run}))


def _entries_for(run=None):
    out = []
    for doc_id, payload in iter_documents():
        raw = _find_text(payload)
        if raw is None:
            continue
        env, body = _parse_envelope(raw)
        if env is None:
            continue
        if run and env.get("run_id") != run:
            continue
        out.append({"id": doc_id, "env": env, "text": body})
    return out


def cmd_list(a):
    for e in sorted(_entries_for(a.run), key=lambda x: (x["env"].get("step", 0), x["env"].get("created_at", ""))):
        env = e["env"]
        print(f'{e["id"][:12]}  step={env.get("step")} {str(env.get("epistemic", "?")):8s} '
              f'agent={str(env.get("agent", "?")):12s} {e["text"][:100]}')


def cmd_export(a):
    for e in _entries_for(a.run):
        print(json.dumps(e))


def cmd_delete(a):
    ids = [s for s in a.ids.split(",") if s]
    failures = 0
    for i in ids:
        try:
            _req("DELETE", f"/api/datasets/{DATASET}/documents/{i}")
            print(f"deleted {i}")
        except urllib.error.URLError as ex:
            failures += 1
            print(f"FAILED {i}: {ex}", file=sys.stderr)
    if failures:
        sys.exit(1)


def cmd_reap(a):
    now = datetime.now(timezone.utc)
    entries = _entries_for()
    superseded = set()
    for e in entries:
        superseded.update(_supersedes_of(e["env"]))
    to_delete = []
    skipped = 0
    for e in entries:
        env = e["env"]
        # Supersede is an explicit instruction from a live writer, so it wins
        # even when this entry's own expiry is malformed.
        if e["id"] in superseded:
            to_delete.append((e["id"], "superseded"))
            continue
        created = _parse_created_at(env.get("created_at"))
        # Missing ttl_hours => fall back to the sweep default; an explicitly
        # malformed value is NOT guessed (fail-closed) but never aborts the sweep.
        ttl = float(a.ttl_hours) if "ttl_hours" not in env else _ttl_hours(env.get("ttl_hours"), a.ttl_hours)
        if created is None or ttl is None:
            skipped += 1
            continue  # unparseable/out-of-range expiry -> KEEP (invariant 1)
        try:
            expires = created + timedelta(hours=ttl)
        except (OverflowError, ValueError):
            # e.g. created_at 9999-12-31 + a TTL that overflows datetime.max (N1)
            skipped += 1
            continue
        if expires < now:
            to_delete.append((e["id"], "expired"))
    print(f"entries={len(entries)} to_delete={len(to_delete)} skipped_unparseable={skipped}")
    if a.dry_run or os.environ.get("STMM_DRY_RUN") == "1":
        for i, why in to_delete:
            print(f"DRY_RUN would delete {i[:12]} ({why})")
        return
    failures = 0
    for i, why in to_delete:
        try:
            _req("DELETE", f"/api/datasets/{DATASET}/documents/{i}")
            print(f"deleted {i[:12]} ({why})")
        except urllib.error.URLError as ex:  # parent of HTTPError (F2)
            failures += 1
            print(f"FAILED {i[:12]} ({why}): {ex}", file=sys.stderr)
    if failures:
        print(f"{failures} delete(s) failed; remaining entries will be retried next sweep", file=sys.stderr)
        sys.exit(1)


def main():
    if not BASE:
        sys.exit("STMM_BASE_URL is required")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--run", required=True); w.add_argument("--agent", required=True)
    w.add_argument("--step", type=int, default=0); w.add_argument("--text", required=True)
    w.add_argument("--epistemic", choices=["observed", "inferred"], default="observed")
    w.add_argument("--ttl-hours", type=int, default=_env_ttl_default())
    w.add_argument("--supersedes", default="")
    w.set_defaults(fn=cmd_write)
    l = sub.add_parser("list"); l.add_argument("--run", required=True); l.set_defaults(fn=cmd_list)
    x = sub.add_parser("export"); x.add_argument("--run", required=True); x.set_defaults(fn=cmd_export)
    r = sub.add_parser("reap"); r.add_argument("--ttl-hours", type=int, default=24)
    r.add_argument("--dry-run", action="store_true"); r.set_defaults(fn=cmd_reap)
    d = sub.add_parser("delete"); d.add_argument("--ids", required=True); d.set_defaults(fn=cmd_delete)
    e = sub.add_parser("ensure-dataset"); e.set_defaults(fn=lambda a: sys.exit(cmd_ensure(a)))
    args = p.parse_args()
    try:
        args.fn(args)
    except urllib.error.URLError as ex:
        # Transport/HTTP failure anywhere (listing, write, delete) surfaces as a
        # single stderr line with a non-zero exit -- never a traceback (N2).
        sys.exit(f"stmem: {ex}")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

