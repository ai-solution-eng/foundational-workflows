# STM interface freeze (Lead-owned)

Frozen 2026-10-14 before any implementer spawn, per PLAN section 6.1
("freeze the interface signatures BEFORE spawning").

## 0. What the Lead already wrote (do NOT rewrite; import it)

src/multimodal_rag/stm_lifetime.py -- the pure lifetime algebra.  Dependency
free, no I/O, no clock.  It is the safety crux and it is FROZEN:

    validate_lifetime(value, *, field)        -> int | None   (raises on bad input)
    validate_stored_lifetime(value, *, field) -> int | None   (never raises; fail-closed)
    effective_lifetime(point, dataset)        -> int | None   (min, None = INFINITY)
    compute_expires_at(created_at, lifetime)  -> int | None
    stamp_point_times(*, now, point_lifetime, dataset_lifetime) -> dict
    reap_decision(payload, dataset_lifetime, now) -> ReapDecision(reap, reason, expires_at)
    dataset_point_reapable(meta) -> bool      (Phase 1 gate)
    dataset_deletable(meta)      -> bool      (Phase 2 gate, invariant 6)

Ownership rule: if a workstream needs a CHANGE to this module, it reports the
need to the Lead.  It does not edit the file.

## 1. Frozen signatures

### T1 -- DatasetManager (dataset_manager.py)

    create_dataset(..., lifetime: int | None = None, kind: str | None = None)
        -> dict          # returned meta carries "lifetime" and "kind"
    delete_dataset(name) -> None            # UNCHANGED signature, no password arg
    list_datasets(...)   -> list | dict     # rows carry "lifetime" and "kind"

Validation: call stm_lifetime.validate_lifetime(lifetime) and let it raise.
Persist BOTH keys into meta.json.  Omit "kind" when None (absent = normal).

### T2 -- RagSystem (rag_system.py)

Ingest accepts a per-document "lifetime" key alongside "text" (metadata dicts
already pass through rag_system.py:1855).  At ingest the server computes:

    stamp = stm_lifetime.stamp_point_times(
        now=<server clock>,
        point_lifetime=<the document's lifetime, if any>,
        dataset_lifetime=<the target dataset's meta lifetime>,
    )

and merges stamp into the document metadata.  The client NEVER supplies
created_at or expires_at.

### T3 -- reaper (NEW src/multimodal_rag/stm_reaper.py + helm + values)

    reap_dataset(dm, name, now) -> ReapResult     # points only, Phase 1
    reap_once(dm, now=None)     -> ReapReport     # sweep every eligible dataset

Reaper contract:
  - dataset selection goes through stm_lifetime.dataset_point_reapable(meta);
    ineligible datasets are skipped WITHOUT being scrolled;
  - the Qdrant Range filter on metadata.expires_at is the CANDIDATE selector;
    stm_lifetime.reap_decision is the AUTHORITY and is re-checked per point;
  - one malformed point must NOT abort the sweep (invariant 3);
  - a failed delete must NOT abandon the remaining deletes (invariant 5);
  - the sweep must terminate even against a repeating pagination cursor
    (invariant 4) -- track seen offsets.

### T4 -- surface (api_server.py + mcp_server.py)

    create_short_term_memory(session_id, ttl_seconds=None, name=None)
    end_short_term_memory(dataset_name)

Both are thin wrappers over the frozen T1 signatures.  The minted name is

    stm-<base32(hmac_sha256(salt, session_id))[:16]>

with the salt from chart values / env.  Catalog hiding filters kind ==
"short_term" in the _rag_acl_filter_datasets(catalog=True) path ONLY.

## 2. The password question -- RESOLVED by the Lead (machine-checked)

ANSWER: the reaper does NOT need a dataset password, and STM datasets must be
created WITHOUT one.  Evidence:

  - api_server.py:716  _require_dataset_password is an API-BOUNDARY gate.
    Lines 734-735 return immediately when dm.has_password(name) is False.
  - dataset_manager.py:2136 delete_dataset(name) takes NO password argument.
    dataset_manager.py:4968 delete_session_history(...) likewise.  The
    password is an HTTP-layer concept, not a storage-layer one.
  - helm-scale-large/templates/backup-cronjob.yaml:71 skips protected
    datasets by design -- that is the SECTION 5 GATE, a separate problem
    (andrew-memory is not backed up), NOT the reaper's problem.
  - stmem/k8s/cronjob-stmem-reaper.yaml:39,106 already carries an OPTIONAL
    STMM_PASSWORD, sent only if set -- the OLD API-driven reaper needed it
    because it drove the REST surface, which enforces the gate.

Consequences for this build:
  1. create_short_term_memory MUST NOT accept or set a password.
  2. If an STM dataset somehow HAS a password, the reaper must SKIP it and
     report it (fail-closed), never attempt a bypass.
  3. Invariant 7 holds: no password bypass is introduced anywhere.
  4. The reaper reads dataset meta through DatasetManager (in-process), not
     through the REST surface, so the gate is never in its path.

## 2b. AMENDMENT -- the stored payload shape (arbitrated by the Lead)

A contract mismatch surfaced between the reaper and the frozen algebra. The
Lead settled it with machine evidence, and the answer is NESTED:

    payload = {"page_content": <text>, "metadata": {<all doc metadata>}}

Evidence: rag_system.py:2650-2657 builds the PointStruct with exactly that
shape, and EVERY pre-existing filter keys on the nested path --
vector_store.py:120/124/128/145 (metadata.file_type, metadata.severities,
metadata.source, metadata.timestamp_start) and dataset_manager.py:4993-4994
(metadata.memory_kind, metadata.session_id).

Therefore:

  * T2 writes the lifetime stamp into the DOCUMENT METADATA, which lands under
    payload["metadata"];
  * the reaper's candidate Range filter key "metadata.expires_at" is correct;
  * the reaper must pass the METADATA MAPPING to reap_decision -- i.e.
    payload["metadata"] when present, falling back to the payload itself for a
    flat/legacy shape.  reap_decision reads "expires_at" at the top level of
    whatever mapping it is handed, and that contract is now explicit.

Symptom of getting this wrong: fail-CLOSED but INERT. The filter selects
candidates and the authority keeps 100% of them, so the feature silently never
reaps anything.

## 2c. AMENDMENT -- helm env wiring (Lead-owned, done)

The API and MCP containers both read the shortTermMemory values as env vars.
The Lead made this edit in helm-scale-large/templates/deployment.yaml (both
env blocks) because that file was outside every teammate's write scope:

    shortTermMemory.enabled         -> gates the whole block
    shortTermMemory.nameSalt        -> RAG_STM_NAME_SALT
    shortTermMemory.defaultTtlHours -> RAG_STM_DEFAULT_TTL_HOURS
    shortTermMemory.maxTtlHours     -> RAG_STM_MAX_TTL_HOURS

An unset salt makes the create/end routes refuse loudly (503) rather than mint
guessable names.

## 2d. AMENDMENT -- the file-ingest cap bypass (Lead-found, under-called by review)

Only ONE of the 17 add_to_vector_store call sites passed dataset_lifetime
(dataset_manager.py:2816, add_documents). The other 16 -- add_files_batch
(:3018) and every typed file handler (:3871-:4125) -- defaulted it to None,
and there is no instance-level fallback in rag_system.py.

Consequence: a document uploaded as a FILE into a finite-lifetime dataset was
stamped expires_at=None, i.e. IMMORTAL. That is a permanent sample inside a
finite dataset, which PLAN section 2.2 row 4 forbids outright ("dataset is a
hard cap -- no permanent samples in a finite dataset").

Worth recording: the XV-1 factual review OBSERVED this mechanism and
classified it "by design not a bug". The Lead overruled that, on the ground
that the ratified truth table makes the cap unconditional. A reviewer
confirming a mechanism is not the same as a reviewer checking it against the
spec -- the spec is the authority, and it says cap.

## 2e. XV-1 execution review findings (all Lead-verified before routing)

The cross-model execution review ran the surface and found 8 defects, each
reproduced by the Lead. The three that matter most:

  F1  MCP create did NOT enforce the D15 "*" create grant that REST enforced.
      A named-ACL key could mint datasets via MCP but not via REST. RULING:
      creating a dataset is a manage act; the gate belongs on BOTH surfaces.
      (The rule: a gate on one surface is not a gate.)
  F4  MCP end checked ownership BEFORE existence, so ending a nonexistent
      dataset returned a FALSE ownership message.
  F5  defaultTtlHours was never clamped to maxTtlHours: a deployment with
      default=48h / max=24h minted 48h datasets on every no-ttl create.

F7 is the one to remember: FOUR mutations survived the T4 test suite because
two assertions had no teeth -- a literal tautology
("assert x if False else True") and an inspect.getsource TEXT match that
passes regardless of behaviour. F1 and F4 lived in exactly that untested
band. A test that asserts a string exists is not a test that the code works.

## 2f. FINAL STATE (Lead-verified)

All 9 tasks complete. 763 passed / 1 skipped on tests/security (from a 707
baseline before this wave). helm lint clean; the reaper CronJob renders 0
objects when disabled and 3 when enabled.

Section 5 Phase 1 acceptance, run against the LIVE cluster's real dataset
meta (14 datasets, 8818 documents, including the four password-protected
memory datasets and andrew-memory's 802 docs): NONE reachable by either
reaper gate. That is the ratified safety claim, checked against production
data rather than a fixture.

The 409 name-echo (F6) was found by the execution reviewer, fixed by the Lead
on BOTH surfaces, and pinned by mutation on both -- reverting either branch
turns a test RED. Accepted residual, stated precisely: a "*"-granted key can
still learn that a given session_id is already taken. The nameless message
removes the NAME, not the fact of collision; per-caller salting would break
the "same session_id -> same dataset" property the design wants.

## 2g. FINAL STATE (Lead-verified, all writers stopped)

  tests/security/    765 passed, 1 skipped   (707 baseline before this wave)
  tests/full_pipeline 474 passed
  helm lint          0 failed; reaper renders 0 objects disabled / 3 enabled
  Section 5 Phase 1  run against the LIVE cluster's real meta (14 datasets,
                     8818 documents incl. the 4 password-protected memory
                     datasets and andrew-memory's 802 docs): NONE reachable by
                     either reaper gate.
  user work          0 stashes; 46 changed paths preserved untouched.

The last review found the wave-1 cap fix was CORRECT BUT UNPINNED: three
mutations that reproduce the original bypass all stayed green, because no test
exercised file ingest and the cap parameter still defaulted to None. Closed
with BOTH a structural guard and an end-to-end pin:

  * _CAP_REQUIRED sentinel -- dataset_lifetime on _add_file_processed is now
    required-by-sentinel, so a forgotten cap RAISES instead of silently
    storing immortal points.  None is a legal lifetime (immortal dataset), so
    it could never double as "caller forgot".
  * test_file_ingest_applies_the_dataset_cap -- asserts the sentinel default
    AND that every add_to_vector_store call site in the module carries the cap.
  * test_file_ingest_cap_guard_raises_when_forgotten -- the loud-failure pin.

Mutation-verified: restoring the None default -> 2 failed; de-capping one leaf
-> 1 failed.  Both previously survived.

## 2h. SEPARATE FINDING -- the "admin" sentinel-name ownership collision (PRE-EXISTING)

Found while investigating the SSO create refusal. NOT caused by the STM work, and
NOT yet fixed -- reported here so it is not lost.

_creator_name_for_request() stamps an ADMIN identity as the literal string
"admin" (api_server.py:2749). Meanwhile "admin" is a perfectly legal CLIENT
name: admin_registry._NAME_RE accepts it (valid_name("admin") is True), so a
registry key or an OIDC user with preferred_username="admin" resolves to a
client identity whose name is "admin".

_identity_may_manage_dataset (the D23 owner-or-admin gate) compares the
caller's NAME to the dataset's created_by stamp. So that client matches
EVERY admin-created dataset.

PROVEN at the gate (stub DatasetManager, real identity resolution):
  client named 'admin' vs a dataset with created_by='admin' -> PASSES
  alice vs the same dataset                                  -> refused (correct)
  client named 'admin' vs alice's dataset                    -> refused (correct)

Reachability today: the D15 ACL gate and the D26 write gate sit in front of
the destructive routes, so a client named "admin" needs an rw grant on the
target to actually delete it. With one, the delete would pass the ownership
check that is supposed to stop it. It is a LATENT hole, not a remote exploit.

THIS MUST BE CLOSED BEFORE ownership becomes an access source (the requested
"grant them access to their own datasets" change), because that change would
turn the same collision into read+write on every admin-created dataset.

## 2i. SSO dataset creation + creator access (product-owner request, 2026-10-14)

TWO defects fixed, plus a security hole found on the way.

D1 -- ordinary authenticated humans could not create a dataset. The gate required
the operator wildcard "*", and an SSO/OIDC identity resolves with an EMPTY dataset
set (dataset_acls().get(name, {})), so every ordinary user was refused 403 AND the
SPA greyed out Create from the same rule.

D2 -- even where creation succeeded, the CREATOR was in no access source, so they
could not read or search the dataset they had just made.

S1 -- PRE-EXISTING, found while investigating: the string "admin" is both the
created_by stamp for admin-created datasets AND a legal client name, so a client
named "admin" passed the D23 owner-or-admin gate on every admin-created dataset.
Latent (the ACL and write gates sit in front) but the D2 change would have turned
it into read+write on every admin dataset. Closed in three layers: reserved names
refused at parse time (warn-and-refuse the ENTRY -- not a raise, because
parse_clients runs at MODULE IMPORT and a raise would be a crashlooping pod),
refused in _validate_name/valid_name (which also covers the OIDC alias and the JWT
identity claim), and refused inside identity_owns_dataset itself so no future
caller can re-open it by comparing names directly.

THE CONFIGURABLE-CREATION DESIGN (product owner, this session):
  1. security.datasetsCreateAllowed -> RAG_DATASETS_CREATE_ALLOWED, default TRUE.
  2. per-user override on the /access page, beating the default in either
     direction, riding the existing flag-only PATCH /api/admin/clients/{name}.
  3. creator access: ownership is an access source, so a creator reaches their own
     dataset and nobody else does.
Three levels verified end to end: default-allow; knob-off denies; per-user
override beats the knob in BOTH directions.

LEAD-OWNED CHART WORK (all three charts, parity): the value and its ConfigMap
wiring. One real bug caught there: '| default true' in sprig treats FALSE as
empty, so the knob rendered "true" even when set false -- the operator could not
turn it off. Removed the default filter; the chart values always define the key.
Verified: default "true", --set false "false", in helm/, helm-scale-medium/ and
helm-scale-large/.

## 3. Open item flagged to the product owner

PLAN section 2.2 row 5 ("point 1h in an immortal dataset -> 1h") is STORED but
NOT ENFORCED, because section 4.5 and invariant 1 both require skipping a
dataset with no lifetime.  The Lead implemented the conservative reading (skip
the dataset; still stamp the point's expires_at so the value is truthful).  If
the owner wants row 5 enforced, the reaper's dataset gate is the one line to
change -- and section 5's Phase 1 acceptance test would need rewriting, since
it is stated in terms of that same skip.

## 4. Standing rules for every teammate

  - Never run git checkout / restore / clean / stash -- the user has
    substantial uncommitted work (37 modified paths + untracked stmem/).
  - Timeout-wrap every test run: timeout 120 ...
  - Exactly one writer per file.  Your write scope is on your task; do not
    write outside it.
  - Mutation-check every test: re-introduce the bug and confirm RED.
