#!/usr/bin/env python3
"""agent_client.py — the client swarm agents use to talk to ST-MEM.

stdlib only. Complements stmem.py (ops CLI + reaper); this module is what an
agent imports (or shells out to) inside a run. Wire format is the stmm/v1
envelope contract from stmem/DESIGN.md: the FIRST line of each stored document
is a JSON envelope, the remainder is the fact text.

Env (same vars as stmem.py):
  STMM_BASE_URL   e.g. http://mm-rag-api.mm-rag.svc:8000   (required for I/O)
  STMM_API_KEY    sent as X-API-Key if set
  STMM_PASSWORD   sent as X-Dataset-Password if set
  STMM_DATASET    default: swarm-stmem
  ST_TTL_HOURS    default ttl for write_fact if not passed (default: 24)

Pure functions (build_envelope, encode_entry, parse_envelope, entry_from_doc,
format_brief) do no I/O and are unit-testable offline; everything that touches
the network goes through _req/_iter_documents.

CLI:
  python3 stmem/agent_client.py brief   --run R [--agent A] [--max-entries N]
  python3 stmem/agent_client.py recall  --run R
  python3 stmem/agent_client.py write   --run R --agent A --text "..." [--step N]
                                        [--epistemic observed|inferred]
                                        [--ttl-hours H] [--supersedes id1,id2]
  python3 stmem/agent_client.py promote --run R        # observed candidates as JSONL
  python3 stmem/agent_client.py new-run-id
  python3 stmem/agent_client.py self-check             # offline pure-function check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone

class StmemError(Exception):
    """Catchable client error: bad config, unreachable server, or HTTP error.

    Library callers should catch this. The CLI converts it to a one-line
    stderr message and exit code 1 (no traceback).
    """

SCHEMA = "stmm/v1"
DEFAULT_DATASET = "swarm-stmem"
DEFAULT_TTL_HOURS = 24
EPISTEMIC_VALUES = ("observed", "inferred")
CREATED_AT_FMT = "%Y-%m-%dT%H:%M:%SZ"  # matches the reaper's parser in stmem.py

# Bounded-output knobs for the handoff brief.
DEFAULT_MAX_ENTRIES = 60
DEFAULT_MAX_FACT_CHARS = 400


# ---------------------------------------------------------------- config (I/O)

def _config():
    """Read env lazily so importing this module has no side effects."""
    base = os.environ.get("STMM_BASE_URL", "").rstrip("/")
    raw_ttl = os.environ.get("ST_TTL_HOURS")
    try:
        ttl = int(raw_ttl) if raw_ttl is not None else DEFAULT_TTL_HOURS
    except ValueError:
        # Explicit failure, never a silent fallback to 24: a silently wrong
        # ttl changes when the reaper deletes facts — fail loudly instead.
        raise StmemError(f"ST_TTL_HOURS must be an integer, got {raw_ttl!r}") from None
    return {
        "base": base,
        "dataset": os.environ.get("STMM_DATASET", DEFAULT_DATASET),
        "api_key": os.environ.get("STMM_API_KEY"),
        "password": os.environ.get("STMM_PASSWORD"),
        "ttl": ttl,
    }


# ------------------------------------------------------- pure: envelope (wire)

def new_run_id() -> str:
    """Stable, unique, human-sortable run id minted by the orchestrator.

    'Stable': derived from wall-clock + random suffix, safe to persist and
    re-print; uniqueness comes from the uuid4 tail.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"stmm-{stamp}-{uuid.uuid4().hex[:8]}"


def build_envelope(run_id, agent, step=0, epistemic="observed", ttl_hours=DEFAULT_TTL_HOURS,
                   supersedes=(), created_at=None):
    """Return the stmm/v1 envelope dict (pure)."""
    if epistemic not in EPISTEMIC_VALUES:
        raise ValueError(f"epistemic must be one of {EPISTEMIC_VALUES}, got {epistemic!r}")
    return {
        "stmm": 1,
        "schema": SCHEMA,
        "run_id": run_id,
        "agent": agent,
        "step": int(step),
        "epistemic": epistemic,
        "created_at": created_at or datetime.now(timezone.utc).strftime(CREATED_AT_FMT),
        "ttl_hours": int(ttl_hours),
        "supersedes": list(supersedes),
    }


def encode_entry(envelope, fact_text) -> str:
    """Envelope dict + fact text -> the exact wire text (pure).

    Contract: first line is the JSON envelope, then a newline, then the fact.
    """
    return json.dumps(envelope) + "\n" + fact_text


def parse_envelope(text):
    """Wire text -> (envelope_dict_or_None, fact_text). Pure.

    No parseable stmm envelope -> (None, text). Callers must treat None as
    'not an stmm entry' (the reaper invariant: unparseable = keep).
    """
    if not isinstance(text, str):
        return None, text
    first, _, rest = text.partition("\n")
    try:
        env = json.loads(first.lstrip())
    except (json.JSONDecodeError, ValueError):
        return None, text
    if not isinstance(env, dict) or env.get("stmm") != 1 or env.get("schema") != SCHEMA:
        return None, text
    return env, rest


def entry_from_doc(doc_id, payload):
    """Storage document -> flat entry dict, or None if not an stmm entry (pure).

    Payload layout varies by write path, so scan for the envelope string.
    """
    raw = None
    if isinstance(payload, str):
        raw = payload
    elif isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, str) and '"stmm"' in v:
                raw = v
                break
    if raw is None:
        return None
    env, fact = parse_envelope(raw)
    if env is None:
        return None
    return {
        "id": doc_id,
        "run_id": env.get("run_id"),
        "agent": env.get("agent"),
        "step": env.get("step", 0),
        "epistemic": env.get("epistemic"),
        "created_at": env.get("created_at"),
        "ttl_hours": env.get("ttl_hours"),
        "supersedes": env.get("supersedes") or [],
        "text": fact,
    }


def _entry_sort_key(entry):
    """Order by (step, created_at) per the DESIGN.md read contract (pure)."""
    return (entry.get("step") or 0, entry.get("created_at") or "")


# --------------------------------------------------- pure: brief formatting

def format_brief(run_id, entries, max_entries=DEFAULT_MAX_ENTRIES, max_fact_chars=DEFAULT_MAX_FACT_CHARS):
    """Compact, prompt-injectable handoff brief (pure).

    Observed facts first (they outrank hypotheses); inferred entries are
    clearly flagged and listed after, per the DESIGN.md anti-poisoning rule.
    Output is bounded: at most max_entries facts, each fact text truncated to
    max_fact_chars. When the bound bites, the NEWEST entries are kept — a
    handoff must reflect the current state of the run, not its beginning —
    and the truncation notice says exactly that (oldest entries omitted).
    Display priority is preserved within what is shown: observed facts are
    listed before inferred ones (anti-poisoning, DESIGN.md).
    """
    ordered = sorted(entries, key=_entry_sort_key)
    total = len(ordered)
    if total > max_entries:
        shown = ordered[-max_entries:]      # newest win the handoff
        dropped_n = total - max_entries     # oldest are dropped
    else:
        shown = ordered
        dropped_n = 0
    obs_show = [e for e in shown if e.get("epistemic") == "observed"]
    inf_show = [e for e in shown if e.get("epistemic") != "observed"]

    lines = [f"# ST-MEM handoff brief — run {run_id}"]
    lines.append(f"# {total} entr{'y' if total == 1 else 'ies'} in scratch memory"
                 f"{f', showing the {len(shown)} NEWEST' if dropped_n else ''}."
                 " Facts below are per-run scratch memory and expire (default 24h).")
    if dropped_n:
        lines.append(f"# TRUNCATED: {dropped_n} OLDEST entr{'y' if dropped_n == 1 else 'ies'} omitted "
                     f"(max_entries={max_entries}); newest facts kept. "
                     "Use recall for the full ordered list.")

    def emit(header, group, tag):
        if not group:
            return
        lines.append(f"## {header}")
        for e in group:
            fact = " ".join((e.get("text") or "").split())
            if len(fact) > max_fact_chars:
                fact = fact[:max_fact_chars].rstrip() + "…"
            lines.append(f"- [{tag} step={e.get('step')} by {e.get('agent') or 'unknown'}] {fact}")

    emit("Observed facts (directly read from tools/outputs)", obs_show, "observed")
    emit("Inferred — HYPOTHESES, weigh below observed (anti-poisoning)", inf_show, "INFERRED")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- I/O layer

def _req(cfg, method, path, body=None):
    url = f"{cfg['base']}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if cfg["api_key"]:
        req.add_header("X-API-Key", cfg["api_key"])
    if cfg["password"]:
        req.add_header("X-Dataset-Password", cfg["password"])
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as e:
        # HTTPError is a URLError subclass; catch it first for the status code.
        detail = ""
        try:
            detail = e.read().decode(errors="replace")[:200]
        except Exception:
            pass
        raise StmemError(f"HTTP {e.code} {method} {path}: {e.reason}"
                         + (f": {detail}" if detail else "")) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise StmemError(f"cannot reach {cfg['base']}: {reason}") from None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise StmemError(f"non-JSON response from {method} {path}: {e}") from None


def _require_base(cfg):
    # Missing config is a normal, user-fixable condition: raise a catchable
    # StmemError (NOT sys.exit — BaseException escapes a caller's except
    # Exception). The CLI turns it into exit 1 + one stderr line.
    if not cfg["base"]:
        raise StmemError("STMM_BASE_URL is required (e.g. http://mm-rag-api.mm-rag.svc:8000)")


def _iter_documents(cfg):
    """Yield (doc_id, payload) across all cursor pages of the stmem dataset.

    Always passes cursor="" (the documented start-of-listing token) so the
    server returns the {documents, next_cursor} page shape.
    """
    cursor = ""
    while True:
        path = (f"/api/datasets/{cfg['dataset']}/documents"
                f"?limit=500&cursor={urllib.parse.quote(cursor)}")
        resp = _req(cfg, "GET", path)
        for doc in resp.get("documents", []):
            yield doc.get("id"), doc.get("payload")
        cursor = resp.get("next_cursor") or ""
        if not cursor:
            return


def write_fact(text, run_id, agent, step=0, epistemic="observed", ttl_hours=None,
               supersedes=()) -> str:
    """Append one fact to scratch memory; returns the stored document id."""
    cfg = _config()
    _require_base(cfg)
    if ttl_hours is None:
        ttl_hours = cfg["ttl"]
    envelope = build_envelope(run_id, agent, step=step, epistemic=epistemic,
                              ttl_hours=ttl_hours, supersedes=supersedes)
    wire = encode_entry(envelope, text)
    resp = _req(cfg, "POST", f"/api/datasets/{cfg['dataset']}/documents", [wire])
    ids = resp.get("stored_ids") or []
    if not ids:
        raise RuntimeError(f"server stored nothing for run {run_id}: {resp}")
    return ids[0]


def recall(run_id):
    """All entries for run_id, ordered by (step, created_at)."""
    cfg = _config()
    _require_base(cfg)
    out = []
    for doc_id, payload in _iter_documents(cfg):
        entry = entry_from_doc(doc_id, payload)
        if entry and entry.get("run_id") == run_id:
            out.append(entry)
    out.sort(key=_entry_sort_key)
    return out


def handoff_brief(run_id, max_entries=DEFAULT_MAX_ENTRIES, max_fact_chars=DEFAULT_MAX_FACT_CHARS) -> str:
    """Prompt-injectable brief for the next agent in the swarm."""
    return format_brief(run_id, recall(run_id), max_entries=max_entries,
                        max_fact_chars=max_fact_chars)


def promote_candidates(run_id):
    """Observed-only entries for the distiller to consider at run end.

    Promotion itself is the distiller's job: re-write durable facts into
    long-term memory via the normal memory path (no stmm envelope), then
    delete the scratch entries. This module never promotes by itself.
    """
    return [e for e in recall(run_id) if e.get("epistemic") == "observed"]


# --------------------------------------------------------------------- CLI

def _self_check():
    """Offline exercise of every pure function. No network, no env needed."""
    env = build_envelope("run-abc", "retriever", step=3, epistemic="observed",
                         ttl_hours=24, supersedes=["deadbeef"],
                         created_at="2026-10-11T14:00:00Z")
    wire = encode_entry(env, "The corpus contains 12 PDFs about transformers.")
    print("wire text:")
    print(wire)
    parsed_env, fact = parse_envelope(wire)
    assert parsed_env == env, "envelope roundtrip failed"
    assert fact == "The corpus contains 12 PDFs about transformers."
    entry = entry_from_doc("doc-1", {"text": wire})
    assert entry["agent"] == "retriever" and entry["step"] == 3 and entry["supersedes"] == ["deadbeef"]
    assert entry_from_doc("doc-2", "plain text with no envelope") is None
    assert parse_envelope("not json\nbody")[0] is None
    assert parse_envelope('{"stmm":0}\nbody')[0] is None  # wrong stmm version -> not ours

    new_id = new_run_id()
    assert new_id.startswith("stmm-") and len(new_id) > len("stmm-") + 9
    assert new_id != new_run_id(), "run ids must be unique"

    entries = [
        entry_from_doc("a", {"text": encode_entry(
            build_envelope("r", "planner", step=5, epistemic="inferred",
                           created_at="2026-10-11T14:02:00Z"), "user probably wants PDFs only")}),
        entry_from_doc("b", {"text": encode_entry(
            build_envelope("r", "retriever", step=1, epistemic="observed",
                           created_at="2026-10-11T14:01:00Z"), "found 12 PDFs about transformers")}),
    ]
    brief = format_brief("r", entries)
    assert brief.index("Observed facts") < brief.index("Inferred"), "observed must come first"
    assert "found 12 PDFs" in brief and "HYPOTHESES" in brief
    # FIX-1 regression: with a bound, the NEWEST step must survive and the
    # notice must say the OLDEST were omitted (direction + truthful text).
    hundred = [entry_from_doc(str(i), {"text": encode_entry(
        build_envelope("r", "a", step=i, epistemic="observed",
                       created_at=f"2026-10-11T{14 + i // 60:02d}:{i % 60:02d}:00Z"), f"fact {i}")})
        for i in range(1, 101)]
    bounded = format_brief("r", hundred, max_entries=10, max_fact_chars=400)
    assert bounded.count("- [") == 10, f"bound violated: {bounded.count('- [')} bullets"
    assert "step=100 " in bounded, "NEWEST step must survive the bound"
    assert "step=1] " not in bounded and "step=1 " not in bounded, "oldest must be dropped"
    assert "TRUNCATED" in bounded and "OLDEST" in bounded and "newest facts kept" in bounded
    assert "90 OLDEST entries omitted" in bounded

    # Newest-overall wins even across epistemic classes; display stays
    # observed-first within the kept set.
    mixed = hundred[:5] + [entry_from_doc("newest-inf", {"text": encode_entry(
        build_envelope("r", "p", step=200, epistemic="inferred",
                       created_at="2026-10-11T23:00:00Z"), "latest hypothesis")})]
    mbrief = format_brief("r", mixed, max_entries=5, max_fact_chars=400)
    assert "step=200" in mbrief and mbrief.index("Observed facts") < mbrief.index("Inferred")

    # FIX-3: bad ST_TTL_HOURS is an explicit catchable StmemError, no fallback.
    saved_ttl = os.environ.get("ST_TTL_HOURS")
    os.environ["ST_TTL_HOURS"] = "abc"
    try:
        _config()
        raise AssertionError("bad ST_TTL_HOURS must raise StmemError")
    except StmemError:
        pass
    finally:
        if saved_ttl is None:
            os.environ.pop("ST_TTL_HOURS", None)
        else:
            os.environ["ST_TTL_HOURS"] = saved_ttl
    print("self-check OK: build/parse roundtrip, entry extraction, ordering, "
          "observed-first brief, NEWEST-survives bounding + truthful notice, "
          "run-id uniqueness, explicit ST_TTL_HOURS failure")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="ST-MEM agent client (stmm/v1)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("brief", help="prompt-injectable handoff brief for a run")
    b.add_argument("--run", required=True)
    b.add_argument("--agent", default=None, help="optional focus agent (currently informational)")
    b.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES)
    b.add_argument("--max-fact-chars", type=int, default=DEFAULT_MAX_FACT_CHARS)
    b.set_defaults(fn=lambda a: sys.stdout.write(
        handoff_brief(a.run, max_entries=a.max_entries, max_fact_chars=a.max_fact_chars)))

    rc = sub.add_parser("recall", help="JSONL entries for a run, ordered by (step, created_at)")
    rc.add_argument("--run", required=True)
    rc.set_defaults(fn=lambda a: [print(json.dumps(e)) for e in recall(a.run)])

    w = sub.add_parser("write", help="append one fact to the run's scratch memory")
    w.add_argument("--run", required=True)
    w.add_argument("--agent", required=True)
    w.add_argument("--text", required=True)
    w.add_argument("--step", type=int, default=0)
    w.add_argument("--epistemic", choices=list(EPISTEMIC_VALUES), default="observed")
    w.add_argument("--ttl-hours", type=int, default=None)
    w.add_argument("--supersedes", default="", help="comma-separated doc ids this entry replaces")
    w.set_defaults(fn=lambda a: print(json.dumps({
        "stored_id": write_fact(a.text, a.run, a.agent, step=a.step,
                                epistemic=a.epistemic, ttl_hours=a.ttl_hours,
                                supersedes=[s for s in a.supersedes.split(",") if s]),
        "run_id": a.run})))

    pr = sub.add_parser("promote", help="observed-only candidates for the distiller, as JSONL")
    pr.add_argument("--run", required=True)
    pr.set_defaults(fn=lambda a: [print(json.dumps(e)) for e in promote_candidates(a.run)])

    nr = sub.add_parser("new-run-id", help="mint a fresh run id")
    nr.set_defaults(fn=lambda a: print(new_run_id()))

    sc = sub.add_parser("self-check", help="offline pure-function check (no network, no env)")
    sc.set_defaults(fn=lambda a: sys.exit(_self_check()))

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except StmemError as e:
        # One-line stderr, non-zero exit, no traceback — for every failure
        # the client itself detects (config, transport, HTTP).
        print(f"agent_client: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
