# ST-MEM — Short-Term Memory for Agent Swarms

Ephemeral, provenance-tagged, per-run scratch memory backed by the
MultimodalRAG dataset store. Everything here **dies by default** unless it is
explicitly promoted into long-term memory at run end.

Contract & design: see [DESIGN.md](DESIGN.md).
Ops CLI + reaper implementation: [stmem.py](stmem.py).
Agent-facing client: `agent_client.py` (this directory).

## The 4 verbs

| Verb | What it does | Where |
|------|--------------|-------|
| **write** | Append one provenance-tagged fact to the run's scratch space. Append-only — never edit an existing entry; supersede it instead. | `agent_client.py write` / `write_fact()` |
| **read** | Enumerate or recall a run's entries ordered by `(step, created_at)`, or render a prompt-injectable handoff brief. | `agent_client.py recall` / `brief` |
| **promote** | At run end, the distiller re-writes the few durable facts into long-term memory (normal memory path, **no stmm envelope**), then deletes the scratch entries. This is the **only path out** of scratch memory. | `agent_client.py promote` (candidates) + distiller |
| **reap** | Enforce the default delete: a CronJob sweeps every 15 minutes and removes expired entries (`created_at + ttl_hours < now`, default **24 hours**) and superseded ones. | `stmem.py reap` via `k8s/cronjob-stmem-reaper.yaml` |

## The envelope (stmm/v1)

Every entry is a plain text document whose **first line is a JSON envelope**;
the remainder is the fact, readable standalone. The envelope lives inside the
text (not payload metadata) so it survives every write path (REST documents,
MCP `add_memory`/`add_documents`) and any agent can parse it without
schema assumptions about payload layout:

```json
{"stmm":1,"schema":"stmm/v1","run_id":"stmm-20261011T140000Z-1a2b3c4d",
 "agent":"retriever","step":3,"epistemic":"observed",
 "created_at":"2026-10-11T14:00:00Z","ttl_hours":24,"supersedes":[]}
The corpus contains 12 PDFs about transformers.
```

Field notes:

- `run_id` — minted by the orchestrator at run start; the scoping key.
- `agent` — writer identity (provenance: who asserted this).
- `step` — monotonic step number within the run (ordering).
- `epistemic` — `observed` (directly read from a tool/output) or `inferred`
  (the agent's hypothesis). Downstream agents **must** weigh
  `inferred < observed` — this is the anti-poisoning control, and the handoff
  brief enforces it by listing observed facts first and flagging hypotheses.
- `created_at` / `ttl_hours` — expiry basis; `ttl_hours` defaults to **24**.
- `supersedes` — doc ids this entry replaces. Append-only discipline: never
  edit, supersede. The next reaper sweep deletes superseded entries.

## Quickstart for a swarm agent

```bash
export STMM_BASE_URL=http://mm-rag-api.mm-rag.svc:8000
export STMM_API_KEY=...            # optional
export STMM_PASSWORD=...           # optional dataset password
export STMM_DATASET=swarm-stmem    # default
export ST_TTL_HOURS=24             # default ttl for written facts

# 1. mint a run id (orchestrator does this once, then passes it around)
RUN=$(python3 stmem/agent_client.py new-run-id)

# 2. agents write facts as they go
python3 stmem/agent_client.py write --run "$RUN" --agent retriever --step 1 \
    --epistemic observed --text "found 12 PDFs about transformers"
python3 stmem/agent_client.py write --run "$RUN" --agent planner --step 2 \
    --epistemic inferred --text "user probably wants PDFs only"

# 3. the next agent in the swarm reads the handoff
python3 stmem/agent_client.py brief --run "$RUN"

# 4. at run end: promote durable facts, everything else dies with the run
python3 stmem/agent_client.py promote --run "$RUN"   # JSONL candidates for the distiller
```

The same operations are importable from Python (`from agent_client import
write_fact, recall, handoff_brief, promote_candidates, new_run_id`) — see the
module docstring in `agent_client.py`.

## Environment variables

| Variable | Meaning | Default |
|----------|---------|---------|
| `STMM_BASE_URL` | MultimodalRAG API base URL (**required** for any network verb) | — |
| `STMM_API_KEY` | Sent as `X-API-Key` if set | unset |
| `STMM_PASSWORD` | Sent as `X-Dataset-Password` if set | unset |
| `STMM_DATASET` | Dedicated scratch dataset (provisioned once via `POST /api/datasets`) | `swarm-stmem` |
| `ST_TTL_HOURS` | Default `ttl_hours` for facts written without an explicit value. A non-integer value is a hard error (the client refuses to guess a deletion deadline). | `24` |

## Error behaviour

All client failures — missing `STMM_BASE_URL`, unreachable server, non-2xx
HTTP responses, non-JSON replies, invalid `ST_TTL_HOURS` — surface as a
**single one-line message, never a traceback**:

- **CLI**: prints `agent_client: <reason>` to stderr and exits non-zero.
- **Library import** (`from agent_client import …`): raises
  `agent_client.StmemError`, a normal `Exception` subclass your agent can
  catch with `except Exception` (it is **not** `sys.exit`, which would slip
  past such a handler and kill the agent process).

## Deletion is the default: 24 hours

- Every entry carries `ttl_hours` (default **24**).
- The reaper CronJob (`k8s/cronjob-stmem-reaper.yaml`) sweeps every **15
  minutes** and deletes (1) entries whose id appears in another entry's
  `supersedes` and (2) entries where `created_at + ttl_hours < now`.
- The reaper **never** deletes anything without a parseable stmm envelope
  whose expiry condition actually holds — unparseable entries are kept.
- `STMM_DRY_RUN=1` (or `stmem.py reap --dry-run`) reports without deleting.
- Cost is bounded by `24h × write rate`; the store is self-cleaning.
- **Promotion is the only escape hatch.** If a fact must outlive the run, the
  distiller must re-write it into long-term memory before the reaper eats it.

## Files

- `agent_client.py` — agent-facing client (stdlib only). Pure functions
  (`build_envelope`, `encode_entry`, `parse_envelope`, `entry_from_doc`,
  `format_brief`) are separated from I/O and offline-testable via
  `python3 stmem/agent_client.py self-check`.
- `stmem.py` — ops CLI: `write` / `list` / `export` / `reap` / `delete`.
- `k8s/cronjob-stmem-reaper.yaml` — ConfigMap + CronJob that enforces the
  24h delete. Credentials come from an existing Secret; create it out-of-band.
- `DESIGN.md` — the contract (wire format, invariants, reviewer checklist).
