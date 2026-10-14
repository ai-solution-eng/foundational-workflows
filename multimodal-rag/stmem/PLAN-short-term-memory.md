# PLAN — Short-Term Memory for Agent Swarms (platform feature)

**Status:** design ratified in conversation (2026-10-14). NOT implemented.
**Supersedes:** the envelope-in-text approach in this directory (see Appendix A).
**Audience:** a fresh agent session / swarm with no memory of the design discussion.
This document is intentionally self-contained — everything needed to execute is here.

---

## 1. Goal

Give an agent swarm an **ephemeral, session-scoped working memory** that behaves as a
first-class platform concept, not a naming convention:

- a **create-short-term-memory tool** mints a dataset for a given session;
- it has a **lifetime** (default **6 hours**, extendable to **1 day** by default;
  both configurable in chart values, and overridable at create time);
- **every entry** may carry its own lifetime;
- **all data is removed at the end** of the session;
- entries can be **searched and added** like any memory;
- short-term datasets are **hidden from the panel** — reachable only by their exact
  (hard-to-guess) name.

## 2. The lifetime model (RATIFIED — do not redesign)

### 2.1 Rule

```
effective_lifetime(point_lifetime, dataset_lifetime):
    treat None as INFINITY
    return min(point_lifetime, dataset_lifetime)
```

`None` = infinite lifetime (the default, so **all existing data is immortal**).

### 2.2 Truth table

| point | dataset | effective | meaning |
|---|---|---|---|
| None | None | **infinite** | today's behaviour — nothing may ever be deleted |
| None | 6h | 6h | point inherits the dataset cap |
| 1h | 6h | 1h | short point stays short |
| 8h | 6h | 6h | **dataset is a hard cap** — no permanent samples in a finite dataset |
| 1h | None | 1h | point caps itself in an infinite dataset |

> An earlier draft used `max()`. **That was wrong** and was corrected by the
> product owner: the dataset lifetime is a **cap**, so `min()` is the rule.

### 2.3 Where the clock comes from

**Server-side only.** An earlier prototype trusted a client-supplied `created_at`,
which lets a writer with a skewed clock shorten or lengthen its own entries' lives.
The server must stamp the time at ingest.

### 2.4 Storage constraint that shapes the design (IMPORTANT)

Qdrant filters cannot do arithmetic — you cannot express
`created_at + lifetime < now` as a filter. Therefore:

- **store `expires_at`** (server-computed at ingest) so the reaper can do a single
  fast **filter-based bulk delete**;
- **also store `lifetime` and `created_at`** for auditability and recomputation;
- if a **dataset lifetime changes later**, existing points do NOT auto-update — a
  recompute pass is required (see section 5.3). Decide and document this behaviour.

## 3. Verified platform facts (cite these; checked live 2026-10-14)

| Fact | Evidence |
|---|---|
| Documents carry **arbitrary metadata** — any non-`text` key becomes `metadata.<key>` | `src/multimodal_rag/rag_system.py:1855` — `meta = {k: v for k, v in inp.items() if k != "text"}` |
| **Filter-based bulk delete already exists** (the primitive for "clear all expired at once") | `src/multimodal_rag/dataset_manager.py` — `delete_session_history` scrolls a Qdrant `Filter` on `metadata.memory_kind` + `metadata.session_id`, deletes by ID list |
| Dataset meta is a **plain dict** — adding `lifetime` is a field addition, not a migration | `dataset_manager.create_dataset` (`meta = {...}`, `self._write_meta(name, meta)`) |
| The platform has **NO per-document TTL** — existing TTL machinery is for caches only (unlock cache, Redis job keys, 24h dataset-existence cache) | grep over `src/multimodal_rag/` |
| **Creating a dataset requires the `*` ACL grant** | `src/multimodal_rag/api_server.py:1004-1006` |
| **Listings already hide un-granted datasets** (D15) — "hidden from the panel" needs no new mechanism, only *not granting* | `_rag_acl_filter_datasets`; the MCP listing reports `note: N dataset(s) hidden` |
| Admin identity **bypasses `dataset_allowed`** | `utils/access_store.py` — `if getattr(identity, "is_admin", False): return True` |
| Qdrant client is reachable as `rag.vector_store._client`; collections are per-dataset | `dataset_manager.delete_document` / `delete_session_history` |

## 4. Design

### 4.1 Dataset level

- New meta field **`lifetime`** (seconds, `None` = infinite) on `create_dataset`.
- New meta field **`kind`** — `"short_term"` for these datasets (absent = normal).
- `POST /api/datasets` accepts both; the MCP surface gets the create tool (4.3).

### 4.2 Point level

- Ingest accepts `lifetime` (seconds, `None` = inherit) alongside `text`.
- Server computes and stores: `lifetime`, `created_at` (server clock),
  `expires_at = created_at + effective_lifetime` (omit/None when infinite).

### 4.3 New tools (keep the surface SMALL)

Add only what cannot be expressed with existing tools:

1. **`create_short_term_memory(session_id, ttl_seconds=None, name=None)`**
   - mints a **non-guessable name**: `stm-<base32(hmac_sha256(salt, session_id))[:16]>`
     (salt from chart values / env; see 4.6);
   - creates the dataset with `kind="short_term"` and `lifetime=ttl_seconds or default`;
   - **grants the caller** (the creating identity) — no one else;
   - enforces `ttl_seconds <= maxTtlHours*3600`;
   - returns the dataset name (the caller must keep it; it is not listable).
2. **`end_short_term_memory(dataset_name)`** — explicit teardown: delete the dataset
   (owner-or-admin gated). Prefer explicit over implicit for Phase 2 (section 5).

**Do NOT add** search/add tools. `add_memory` / `search_dataset` /
`dataset_add_documents` already work against any dataset the caller can reach —
reuse them with the minted name. (Product owner: "I'm open to the design here.")

### 4.4 Hidden from the panel

- **Do not grant** the dataset to any user key except the creating identity — the
  existing D15 ACL filter already removes it from listings.
- Additionally **filter `kind == "short_term"` out of the catalog/panel view**
  (`_rag_acl_filter_datasets(..., catalog=True)` path) so it does not appear even
  for the creator. Direct-name access still works.
- The hashed name is **defense-in-depth**, not the mechanism.

### 4.5 Reaper (CronJob, configurable frequency)

- One sweep, **bulk filter delete** of expired points across short-term datasets:
  scroll `Filter(must=[FieldCondition(key="metadata.expires_at", range=Range(lt=now))])`
  and delete by ID list — mirroring `delete_session_history`.
- Must handle: datasets with no `lifetime` (skip entirely), points with no
  `expires_at` (skip — immortal), transport failures (do not abandon the sweep).
- Follow the operational lessons in section 8.

### 4.6 Chart values (values-blueprint style — user-readable, configurable first)

```yaml
shortTermMemory:
  enabled: true
  defaultTtlHours: 6        # create-time default
  maxTtlHours: 24           # hard cap on create-time override
  reapSchedule: "*/15 * * * *"
  nameSalt: ""              # REQUIRED when enabled; set per deployment
  deleteDatasetsOnExpiry: false   # PHASE 2 GATE — see section 5
```

## 5. Phased rollout (safety gates are NOT optional)

### Phase 1 — entry lifetimes (SAFE: cannot touch existing data)

Dataset `lifetime` field, point `lifetime` field, `min()` semantics, reaper deleting
**expired points only**.

Why it is safe: every existing dataset has `lifetime = None`, and
`min(None, None) = infinite`, so **existing data is mathematically unreachable** by
the reaper. This is an arithmetic guarantee, not a filter we must remember to write.

**Acceptance:** a test that enumerates real dataset meta and asserts none is
reachable by the reaper's selection predicate.

### Phase 2 — dataset lifecycle (GATED)

`create_short_term_memory`, hidden naming, and **deletion of the dataset itself**
at end-of-life.

**Gate (do not skip):** the dataset-deletion path may only ship once:
1. the backup path is proven end-to-end (a real run + a verified restore), AND
2. there is a decision on **password-protected datasets** — the backup script
   *skips* them by design, so `andrew-memory` (802 docs) is currently **not backed
   up at all**; and
3. the delete-time semantics are settled: **reuse `lifetime` as the delete time, or
   add a separate field** (recommendation: a separate field such as `retain_until`,
   because "when should the entries die" and "when should the container be
   destroyed" are different questions).

**Safety invariants for Phase 2:**
- the reaper may delete a dataset **only** when `kind == "short_term"` **and**
  `lifetime` is set — never a dataset with a missing/None lifetime;
- there is no existing code path that auto-deletes a dataset; this would be the
  first, so it needs its own review;
- prefer **explicit** teardown (`end_short_term_memory`) with automatic expiry as a
  backstop, not the primary mechanism.

### Phase 3 — protected-dataset backups (separate track)

Supply the password to the backup job (or document the gap) so protected datasets
are captured. Independent of this feature but a prerequisite for trusting deletion.

## 6. Swarm organisation

### 6.1 File-ownership rule (hard)

`dataset_manager.py`, `api_server.py`, `rag_system.py`, `mcp_server.py` are large and
shared. **Exactly one writer per file per wave.** Proposed partition:

| Task | Owner scope (write) | Deliverable |
|---|---|---|
| **T1 dataset lifetime** | `dataset_manager.py` (create/meta only) | `lifetime` + `kind` on create; read-back in meta |
| **T2 point lifetime** | `rag_system.py` | accept `lifetime`; server-stamp `created_at`/`expires_at`; expose in payload |
| **T3 reaper + chart** | new module `src/multimodal_rag/stm_reaper.py`, `helm-scale-large/templates/*`, `values*.yaml` | filter-based sweep; CronJob; values |
| **T4 API/MCP surface** | `api_server.py` (create route + catalog filter), `mcp_server.py` | create/end tools; hide short-term from catalog |
| **T5 tests** | `tests/security/test_stm_*.py` | unit + integration + **safety invariants** |

T1–T4 touch different files and may run in parallel; T5 depends on the interfaces
(freeze the interface signatures BEFORE spawning).

### 6.2 Cross-verification protocol (proven in this project)

- **Primary** implements; **the other cell reviews at decision points only**.
- Spawn **durable teammates** (never anonymous background children) with the
  provider and model pinned **together**:
  - `provider: "pcai-model-endpoint-flow"`, `model: "Deepseek-V4.1-Flash"`
  - `provider: "glm53-flash-sproket"`, `model: "glm-5.3-flash"`
  - Verify the spawn result echoes the pinned model — **if it says `auto`, the pin
    did not apply; do not proceed on that teammate.**
- Reviewer prompt shape: *"You are the FACTUAL-INTEGRITY reviewer. For each concrete
  problem: quote the claim, state the correction if confident, else UNVERIFIABLE.
  No style comments. If clean, reply exactly REVIEW-CLEAN."*
- **Machines outrank models**: a dry-run, a log line, or a test result beats any
  model's assertion.

### 6.3 Verification standard (this project's bar)

Every fix must be **mutation-checked**: re-introduce the bug in a throwaway copy and
confirm the test goes RED. A test that passes both with and without the fix pins
nothing. This caught a real regression class in this project.

## 7. Safety invariants (acceptance criteria — test these explicitly)

1. **Existing data is untouchable.** A dataset with no `lifetime` is never selected
   by the reaper, for either point- or dataset-level deletion. Prove by enumeration
   over real meta.
2. **`None` means infinite**, at both levels, and `min()` is the only combination
   rule. Prove with the 2.2 truth table as a parametrized test.
3. **The reaper never deletes without a server-stamped `expires_at`.** Unparseable /
   missing / malformed means KEEP (fail-closed), and a malformed entry must **never**
   abort the sweep (one bad entry must not stop every other entry from being reaped).
4. **The reaper always terminates**, even against a repeating pagination cursor.
5. **A failed delete does not abandon the remaining deletes.**
6. **Phase 2 only:** the dataset-deletion predicate requires `kind == "short_term"`
   AND a set `lifetime`. A normal dataset is never deleted, whatever its fields.
7. **No password bypass is introduced anywhere.** Dataset passwords protect user
   data from the operator; admins must prove them (see Appendix C).

## 8. Operational lessons to carry forward

- **Never run `git checkout` / `restore` / `clean` / `stash`** — this repo has
  substantial uncommitted user work. Ask first.
- **Timeout-wrap every command** (`timeout 120 ...`).
- **Regenerate any embedded copy after editing the source.** The `stmem` CronJob's
  ConfigMap embeds a copy of `stmem.py`; a stale copy silently runs old logic. There
  is a parity checker at `stmem/_lead_checks/verify_manifest.py`.
- **CronJob status is wiped on redeploy.** `kubectl get cronjob` shows empty
  last-run fields after any redeploy; check the artifact (S3/MinIO) instead.
- **UTC is the accepted convention** for schedules (AWS-style). Do not "fix" the
  backup schedule to local time — the product owner explicitly prefers UTC.
- **applygate is fail-closed:** plan, human review, apply with the plan hash;
  `Secret` kinds are refused (create them out-of-band); a refusal is the answer.

## 9. Open questions for the product owner (resolve before Phase 2)

1. **Dataset delete time:** reuse `lifetime`, or a separate `retain_until`?
   (Recommendation: separate.)
2. **Session end:** explicit `end_short_term_memory` call, or purely time-based?
   (Recommendation: explicit primary + expiry backstop.)
3. **Dataset lifetime change:** recompute `expires_at` for existing points
   (migration pass), or leave them at their ingest-time value?
4. **Backups:** should the backup job hold dataset passwords so protected datasets
   (including `andrew-memory`) are actually captured?

## 10. Definition of done

- All section 7 invariants have passing tests, mutation-checked.
- Phase 1 shipped: entry lifetimes live, reaper reaping expired points only,
  existing data provably untouched.
- Phase 2 shipped only if the section 5 gate is satisfied and signed off.
- Chart values documented and configurable (default 6h / max 24h / reap schedule).
- Cross-verification completed by the opposing model cell on each artifact.

---

## Appendix A — what exists today in this directory (and what to do with it)

An earlier prototype implemented short-term memory **without** platform support:

| File | Status |
|---|---|
| `stmem/stmem.py` | envelope-in-text + CLI reaper. **Superseded for storage format** — use real metadata (4.2), not a JSON line in the text. The reaper's robustness logic (fail-closed, non-aborting sweeps, cursor-loop guard, range/overflow guards) is worth **porting**, not discarding. |
| `stmem/agent_client.py` | agent-facing wrapper (`write_fact`/`recall`/`handoff_brief`/`promote_candidates`). The **concepts** (provenance, `epistemic: observed|inferred`, observed-first handoff, promotion-only-exit) are the valuable part and should survive into the platform version. |
| `stmem/tests/test_stmem.py` | 92 tests, mutation-checked. Reuse the test *shapes*; the assertions target the old format. |
| `stmem/k8s/cronjob-stmem-reaper.yaml` | deployed and working (15-min sweep). Phase 1's reaper supersedes it. |
| `stmem/DESIGN.md`, `DEPLOYMENT.md` | historical; keep for the operational lessons. |

**Why the envelope approach was wrong:** `rag_system.py:1855` shows documents carry
arbitrary metadata natively. The envelope was chosen only because that could not be
confirmed at the time — it was a workaround for an unverified assumption. Real
metadata fields are strictly better (queryable, filterable, no parsing, no
chunk-split hazard).

## Appendix C — password/admin rules (do not regress)

- A dataset password protects **the user's data from the operator**: an admin must
  **prove** it, exactly like anyone else. There is a regression test asserting
  `_require_dataset_password` contains **no** `is_admin` bypass — keep it.
- An admin cannot *self-select* (nothing to select into). The supported path is the
  admin **unlock** branch in `api_select_dataset`, which verifies the password and
  warms the unlock cache (`source: "admin-unlock"`).
- `RAG_UNLOCK_MAX_TTL=0` is a **"no-expiry allowed" sentinel, not a zero cap** —
  TTL validation must guard the comparison (`ttl_max and ttl > ttl_max`).
