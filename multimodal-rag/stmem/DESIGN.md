# ST-MEM: Short-Term Memory for Agent Swarms (stmm/v1)

Ephemeral, provenance-tagged, per-run scratch memory backed by the MultimodalRAG
dataset store. Complements long-term memory: everything here dies by default
unless explicitly promoted at run end.

## Contract

**Store**: dedicated dataset (default `swarm-stmem`), provisioned once via
`POST /api/datasets`. Entries are plain text documents.

**Envelope (stmm/v1)**: every entry is self-describing. The FIRST line of the
text is a JSON envelope; the remainder is the fact:

    {"stmm":1,"schema":"stmm/v1","run_id":"...","agent":"...","step":3,
     "epistemic":"observed","created_at":"2026-10-11T14:00:00Z",
     "ttl_hours":24,"supersedes":[]}
    <self-contained fact text, readable standalone>

The envelope is embedded in the text (not payload metadata) deliberately:
it survives every write path (REST documents, MCP add_memory/add_documents)
regardless of which dict keys the storage layer preserves, and any agent can
parse it without schema assumptions about payload layout.

**Fields**
- run_id: minted by the orchestrator at run start; the scoping key.
- agent: writer identity (provenance — who asserted this).
- step: monotonic step number within the run (ordering).
- epistemic: "observed" (directly read from a tool/output) or "inferred"
  (agent's hypothesis). Downstream agents MUST weigh inferred < observed —
  this is the anti-poisoning control.
- created_at / ttl_hours: expiry basis; default 24h.
- supersedes: doc ids this entry replaces. Append-only discipline: never edit,
  supersede. A superseded entry is deleted by the next reaper sweep.

**Writes**: append-only. POST /api/datasets/{name}/documents with the
envelope-prefixed text. No read-modify-write; no mutable "current state" blob.

**Reads**
- Enumeration / handoff snapshot: GET /api/datasets/{name}/documents
  (cursor-paginated), filter envelopes by run_id, order by (step, created_at).
- Semantic recall: search the dataset normally, then validate hits carry a
  stmm envelope with the expected run_id before trusting them.

**Promotion (run end)**: the distiller reviews `export` output and re-writes
the few durable facts into long-term memory (normal memory path, no stmm
envelope, kind=fact/decision). Everything else dies with the run. The
promotion step is the ONLY path out of the scratch store.

**Reaping (default delete = 24h)**: a CronJob sweeps every 15 minutes:
1. delete any entry whose id appears in another entry's `supersedes`;
2. delete any entry where created_at + ttl_hours < now.
Failures are logged, never fatal for other entries. STMM_DRY_RUN=1 reports
without deleting. Cost is bounded by 24h x write rate.

## API endpoints used (verified in src/multimodal_rag/api_server.py)
- POST   /api/datasets                          -> create (409 if exists) [ensure-dataset]
- GET    /api/datasets/{name}/documents         -> {documents:[{id,payload}], next_cursor}
- POST   /api/datasets/{name}/documents         -> {stored_ids:[...]}  (body: JSON array or object)
- DELETE /api/datasets/{name}/documents/{doc_id} -> {status:"ok"}

## Components
- stmem.py       — stdlib-only CLI: write / list / export / reap / delete.
- k8s/cronjob-stmem-reaper.yaml — ConfigMap (reaper script) + CronJob.
                   Credentials come from an existing Secret (STMM_SECRET_NAME,
                   keys STMM_BASE_URL, STMM_API_KEY, STMM_PASSWORD, STMM_DATASET);
                   create it out-of-band — never in this manifest.

## Known limitations (from cross-cell review, 2026-10-11)
- **MCP add_memory is NOT a safe writer for oversized entries** (reproduced
  2026-10-11 against \`_split_memory_text\`): a memory larger than
  MEMORY_MAX_TOKENS with NO markdown headings is split positionally, and only
  chunk 1 carries the envelope — measured: 1 of 32 chunks retained it, leaving
  31 permanently invisible to the reaper. Entries WITH \`## \` headings are safe
  (the header, which holds the envelope, is prepended to every chunk). Since an
  ST-MEM fact is a first line plus body text, it has no headings: keep entries
  comfortably under MEMORY_MAX_TOKENS (default 8192) or write via raw
  POST /documents, which does not split.
- The reaper sweeps the WHOLE dataset with no run_id filter. A doc whose first
  line is a valid envelope is treated as scratch even if it belongs to another
  run. Use a dedicated dataset (default \`swarm-stmem\`) — never a shared one.
- \`X-API-Key\` is accepted by the live tree; \`.baseline_head\` middleware only
  reads \`X-RAG-Api-Key\`/Bearer. stmem.py sends BOTH.
- The CronJob image is tag-pinned (\`python:3.12-alpine\`) — pin by digest before
  production.
- The ConfigMap EMBEDS A COPY of stmem.py. Regenerate the manifest after every
  edit (the lead's generator does this; verify with _lead_checks/verify_manifest.py).

## Verification status (2026-10-11)
- \`stmem/tests/test_stmem.py\` — 92 tests, green; mutation-checked (each pre-fix
  behaviour re-introduced in a throwaway copy turns the matching test RED).
  behaviour re-introduced in a throwaway copy turns the matching test RED).
- \`_lead_checks/it_stmem.py\` — 18 end-to-end checks (write/list/dry-run/reap/fail-closed): ALL PASS.
  API (write/list/dry-run/reap/fail-closed invariants): ALL PASS.
- \`_lead_checks/it_agent_client.py\` — 15 cross-component checks (client writes, frozen reaper reads/deletes): ALL PASS.
  frozen reaper reads and deletes): ALL PASS.
- \`_lead_checks/verify_manifest.py\` — ConfigMap embed == source, CronJob fields consistent: PASS.
- \`_lead_checks/probe_delta.py\` — 29 adversarial checks on the N1/N2 delta (range/overflow
  boundaries, non-JSON and empty 2xx bodies): ALL PASS.
- NOT yet verified live: no entry has been written to or reaped from a real
  \`swarm-stmem\` dataset (the API is in-cluster only and the dataset does not
  exist yet). Deployment is the remaining step.

## Invariants a reviewer must check
1. The reaper NEVER deletes an entry without a parseable stmm envelope whose
   expiry condition actually holds (missing/unparseable envelope = keep).
2. Supersede deletion only fires when the superseding entry exists in the
   same sweep (no dangling supersedes causing premature loss).
3. Append-only: no code path rewrites an existing entry.
4. Promotion is a re-write into long-term memory followed by scratch deletion,
   never an in-place mutation.
