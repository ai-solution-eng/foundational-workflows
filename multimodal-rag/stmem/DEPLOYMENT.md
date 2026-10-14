# ST-MEM deployment runbook

Status: code complete and verified offline. **Not yet deployed.** The only
remaining step is the cluster apply, which needs human sign-off (and one ACL
decision, below).

## What ships

| Artifact | sha256 (frozen 2026-10-11) |
|---|---|
| `stmem/stmem.py` | `sha256:750ed38f042f7df96ac7ac2fc46b9758311a3ababe8f09dc6f2a93f4a9ba6baf` |
| `stmem/agent_client.py` | `sha256:725b596752a04714023af965dd9a9e65846d0aea0ccf68748dafb67c22a669be` |
| `stmem/tests/test_stmem.py` | `sha256:373282d345958395cf34fbeea7ddc024486d2575a4c0a8d0a26d280d56a93c0f` |
| `stmem/k8s/cronjob-stmem-reaper.yaml` | `sha256:7d5e663173088cfc1037e57483fab0c98d0e91476f75fcfc85c2eb85b879d392` |
| `stmem/DESIGN.md` | `sha256:62a26086190a6e3db8e06b2ed3384fc7dcc418682b0a077e491b3a5d36aa2c0a` |

Verification at these hashes: 92 unit tests green (mutation-checked), 18 core
end-to-end checks, 15 cross-component checks, manifest parity check — all PASS.
**No live write to a real `swarm-stmem` dataset has happened yet.**

## Step 1 — decide the ACL path (blocks everything)

Creating a dataset requires the `*` grant on the API key
(`src/multimodal_rag/api_server.py:1004-1006`). Two options:

- **A (broader key)**: give the reaper's key `*`. The init container creates
  `swarm-stmem` on the first sweep. Simplest, but the key can mint datasets.
- **B (pre-create, narrower key)**: create `swarm-stmem` once (as an admin or
  with a `*` key), grant the reaper key access to just that dataset. The init
  container then gets 409 and no-ops. **Preferred** — least privilege.

Note: `/select` self-service cannot bootstrap this, because it can only grant
access to a dataset that already exists.

## Step 2 — create the credentials Secret (out-of-band; applygate refuses Secrets)

```bash
kubectl -n mm-rag create secret generic stmem-reaper-secrets \\
  --from-literal=STMM_BASE_URL=http://rag-mcp-server-api.mm-rag.svc.cluster.local \\
  --from-literal=STMM_API_KEY=<key with * or swarm-stmem access> \\
  --from-literal=STMM_DATASET=swarm-stmem
# add --from-literal=STMM_PASSWORD=... only if the dataset is password-protected
```

## Step 3 — apply the manifest (plan-then-apply, hash-bound)

The plan is already recorded; the apply must carry the matching hash:

    applygate.plan_apply(namespace="mm-rag", manifest=<file bytes>)   # dry-run
    applygate.apply_manifest(namespace="mm-rag", manifest=<same bytes>,
                             confirm_apply=True, plan_sha256=<from plan>)

Editing the file invalidates the plan — re-plan first. This is the intended
fence, not an obstacle.

## Step 4 — verify the first sweep

```bash
kubectl -n mm-rag get cronjob stmem-reaper
kubectl -n mm-rag create job --from=cronjob/stmem-reaper stmem-reaper-manual
kubectl -n mm-rag logs job/stmem-reaper-manual -c ensure-dataset
kubectl -n mm-rag logs job/stmem-reaper-manual -c reaper
```
Expected: `dataset swarm-stmem already exists` (or `created`), then
`entries=0 to_delete=0 skipped_unparseable=0`.

## Step 5 — first real round trip (proves the whole loop)

From inside the cluster (the API is not reachable from a dev host):

```bash
python3 stmem.py write --run smoke-1 --agent lead --step 1 --ttl-hours 0 \\
    --text "smoke test: this entry must be reaped"
python3 stmem.py reap --dry-run     # expect to_delete=1 (expired)
python3 stmem.py reap               # expect deleted ...
python3 stmem.py list --run smoke-1 # expect no rows
```

## Operating notes

- **Regenerating the manifest**: the ConfigMap embeds a copy of `stmem.py`.
  After ANY edit to `stmem.py`, regenerate the manifest and re-plan, then run
  `python3 stmem/_lead_checks/verify_manifest.py` (exits non-zero on drift).
- **Image pinning**: `python:3.12-alpine` is a mutable tag. Pin by digest
  before production.
- **Known limits** (see DESIGN.md "Known limitations"): MCP `add_memory` is not
  a safe writer for oversized entries; the reaper sweeps the whole dataset with
  no run_id filter, so use a DEDICATED dataset; X-API-Key vs X-RAG-Api-Key
  divergence is handled by sending both.
