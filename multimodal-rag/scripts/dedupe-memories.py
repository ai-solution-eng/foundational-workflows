#!/usr/bin/env python3
"""Memory-dataset dedupe: the curation pass that follows the 2026-10-15
gateway duplicate fumbles (cross-replica distill races + the supersede gate
that only matched session_history).

Reads a full-fidelity documents export (GET
/api/datasets/{name}/documents/download?format=jsonl -- one
{"id", "text", "metadata"} object per Qdrant point; carries memory_id,
memory_ts, session_id, memory_kind) and groups duplicates TWO ways:

  1. SAME-SESSION: same session_id + same memory_kind -- exactly what the
     (now deployed) supersede-in-place gate would have done at write time.
     Keeps the newest memory_ts, deletes the older points.
  2. NEAR-IDENTICAL across different sessions within the lookback window:
     the gateway fumbles distilled ONE conversation repeatedly under fresh
     thread ids, so session grouping cannot see them. Detected by
     shingle-Jaccard similarity on the normalized text (default >= 0.70),
     keeping the newest/richest point of each cluster.

Split memories (memory_chunks > 1) are grouped by their FIRST chunk's id
(chunk_index == 0 carries the full decision; chunks share session_id +
memory_ts, and every chunk of a kept memory survives while every chunk of
a deleted one goes).

DRY-RUN BY DEFAULT: prints the deletion manifest (id, reason, kept id) and
writes it to <input>.dedupe.json. --apply performs the deletions over REST
(needs RAG_API_KEY; dataset password via RAG_DATASET_PASSWORD when the
dataset is protected). Deleted ids are exactly the Qdrant point ids the
export carries, so the manifest doubles as an audit record.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_documents(path: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if not doc.get("id"):
                continue
            meta = doc.get("metadata") or {}
            docs.append(
                {
                    "id": str(doc["id"]),
                    "text": doc.get("text") or "",
                    "kind": str(meta.get("memory_kind") or meta.get("kind") or "note"),
                    "session_id": str(meta.get("session_id") or ""),
                    "ts": str(meta.get("memory_ts") or ""),
                    "chunk_index": int(meta.get("chunk_index") or 0),
                    "chunk_total": int(meta.get("memory_chunks") or meta.get("chunk_total") or 1),
                }
            )
    return docs


def parse_ts(ts: str) -> datetime:
    if not ts:
        return datetime.min.replace(tzinfo=UTC)
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _shingles(text: str, size: int = 8) -> set[str]:
    words = _WS.sub(" ", text.lower()).split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

# Point ids already removed by an earlier pass -- the near-identical pass
# must never emit the same deletion twice (the manifest and the DELETE call
# would double-count them, and Qdrant delete is idempotent but the audit
# record should be clean).
HANDLED: set[str] = set()


def _group_same_session_impl(
    docs: list[dict[str, Any]], supersede_kinds: tuple[str, ...]
) -> list[tuple[list[dict[str, Any]], dict[str, Any], str]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        if doc["session_id"] and doc["kind"] in supersede_kinds:
            groups[(doc["kind"], doc["session_id"])].append(doc)
    out: list[tuple[list[dict[str, Any]], dict[str, Any], str]] = []
    for (kind, sid), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        # cluster by (memory_ts, chunk_total): each WRITE produced one
        # multi-chunk batch sharing ts; keep the newest write, drop the rest
        writes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for m in members:
            writes[m["ts"]].append(m)
        if len(writes) < 2:
            continue
        newest_ts = max(writes, key=parse_ts)
        kept = writes[newest_ts]
        dropped = [d for ts, ds in writes.items() if ts != newest_ts for d in ds]
        for d in dropped:
            HANDLED.add(d["id"])
        out.append((dropped, kept[0], f"same session {sid} ({kind}): superseded by {newest_ts}"))
    return out


def group_near_identical(
    docs: list[dict[str, Any]], window_hours: float, threshold: float
) -> list[tuple[list[dict[str, Any]], dict[str, Any], str]]:
    """Cluster near-identical texts (any session) inside the window."""
    cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
    recent = [d for d in docs if parse_ts(d["ts"]) >= cutoff and d["kind"] != "note"]
    # one shingle-set per WRITE (grouped by session+ts so multi-chunk
    # memories compare as a whole)
    writes: dict[tuple[str, str], dict[str, Any]] = {}
    for d in recent:
        key = (d["session_id"], d["ts"])
        w = writes.get(key)
        if w is None:
            w = {"session": d["session_id"], "ts": d["ts"], "parts": [], "docs": []}
            writes[key] = w
        w["parts"].append(d["text"])
        w["docs"].append(d)
    shingled = [
        (k, w, _shingles("\n".join(w["parts"])))
        for k, w in writes.items()
        if not all(d["id"] in HANDLED for d in w["docs"])
    ]

    # greedy union-find over pairs above threshold (small windows: fine)
    parent = {k: k for k, _, _ in shingled}

    def find(x: Any) -> Any:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(shingled)):
        for j in range(i + 1, len(shingled)):
            if jaccard(shingled[i][2], shingled[j][2]) >= threshold:
                a, b = find(shingled[i][0]), find(shingled[j][0])
                if a != b:
                    parent[b] = a

    clusters: dict[Any, list[tuple[Any, dict[str, Any], set[str]]]] = defaultdict(list)
    for entry in shingled:
        clusters[find(entry[0])].append(entry)

    out: list[tuple[list[dict[str, Any]], dict[str, Any], str]] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda e: (parse_ts(e[1]["ts"]), sum(len(t) for t in e[1]["parts"])))
        kept = members[-1][1]
        dropped_docs = [d for e in members[:-1] for d in e[1]["docs"]]
        out.append(
            (
                dropped_docs,
                kept["docs"][0],
                "near-identical (Jaccard>=%.2f) across %d sessions; kept session %s @ %s"
                % (threshold, len({e[1]["session"] for e in members}), kept["session"], kept["ts"]),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Apply (REST) + reporting
# ---------------------------------------------------------------------------

def apply_deletions(
    api: str, dataset: str, ids: list[str], api_key: str, password: str
) -> tuple[int, str]:
    """DELETE /api/datasets/{name}/documents with explicit point ids."""
    url = f"{api.rstrip('/')}/api/datasets/{dataset}/documents"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-RAG-Api-Key"] = api_key
    if password:
        headers["X-Dataset-Password"] = password
    req = urllib.request.Request(
        url, method="DELETE", data=json.dumps({"doc_ids": ids}).encode(), headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode()[:500]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:500]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", help="documents.jsonl export path")
    ap.add_argument("--window-hours", type=float, default=24.0, help="near-dup lookback (default 24)")
    ap.add_argument("--threshold", type=float, default=0.70, help="shingle-Jaccard keep threshold")
    ap.add_argument(
        "--supersede-kinds",
        default="project-summary,session_history",
        help="comma kinds eligible for same-session supersede (default matches the gateway writer)",
    )
    ap.add_argument("--apply", action="store_true", help="perform deletions (default: dry-run)")
    ap.add_argument("--api", default=os.environ.get("RAG_API_URL", ""), help="REST base url for --apply")
    ap.add_argument("--dataset", default="", help="dataset name for --apply")
    args = ap.parse_args()

    docs = load_documents(args.export)
    supersede_kinds = tuple(k.strip() for k in args.supersede_kinds.split(",") if k.strip())
    deletions: list[tuple[list[dict[str, Any]], dict[str, Any], str]] = []
    deletions += _group_same_session_impl(docs, supersede_kinds)
    deletions += group_near_identical(docs, args.window_hours, args.threshold)

    manifest: dict[str, Any] = {
        "generated": datetime.now(UTC).isoformat(),
        "input": args.export,
        "documents": len(docs),
        "window_hours": args.window_hours,
        "threshold": args.threshold,
        "groups": [],
        "delete_ids": [],
    }
    for dropped, kept, reason in deletions:
        manifest["groups"].append(
            {
                "reason": reason,
                "kept": kept["id"],
                "delete": [d["id"] for d in dropped],
            }
        )
        manifest["delete_ids"].extend(d["id"] for d in dropped)

    out_path = args.export + ".dedupe.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"{len(docs)} documents read from {args.export}")
    print(f"{len(deletions)} duplicate group(s), {len(manifest['delete_ids'])} point(s) to delete")
    for group in manifest["groups"]:
        print(f"  KEEP {group['kept']}")
        for did in group["delete"]:
            print(f"    del {did}  -- {group['reason']}")
    print(f"manifest: {out_path}")

    if not args.apply:
        print("dry-run: nothing deleted (pass --apply to execute)")
        return 0

    if not (args.api and args.dataset):
        print("--apply needs --api and --dataset", file=sys.stderr)
        return 2
    status, body = apply_deletions(
        args.api,
        args.dataset,
        manifest["delete_ids"],
        os.environ.get("RAG_API_KEY", ""),
        os.environ.get("RAG_DATASET_PASSWORD", ""),
    )
    print(f"DELETE status {status}: {body}")
    return 0 if status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
