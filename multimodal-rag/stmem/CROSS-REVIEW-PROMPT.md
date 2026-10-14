# DeepSeek cross-review prompt — ST-MEM v0.1

Review the following implementation for correctness against the stated
invariants. Direction-matched posture: the primary (GLM) implementation was
generated in a single pass by one model; the highest-value review targets are
(1) logic errors invisible to the author, (2) fail-open paths where the design
promises fail-closed, (3) concurrency/pagination edge cases the author never
exercised (nothing here has run against a live server yet).

## Files to review
- stmem/DESIGN.md            (contract + invariants)
- stmem/stmem.py             (client + reaper)
- stmem/k8s/cronjob-stmem-reaper.yaml (ConfigMap copy of stmem.py + CronJob)

## Questions to answer explicitly
1. Invariant 1 (never delete without a parseable envelope whose expiry HOLDS):
   trace cmd_reap + _entries_for + _parse_envelope end-to-end. Is there ANY
   path where a doc with no envelope, or an unparseable expiry, gets deleted?
   Is there any path where a VALID expired entry silently survives forever?
2. Supersede semantics: build of the superseded set and the "superseding
   entry exists" check — is the check sufficient? What happens if the
   superseding entry is deleted between sweeps? If the superseded entry is
   ALSO the superseding entry of another chain?
3. iter_documents pagination: cursor="" start, next_cursor loop termination,
   limit=500 vs server le=1000. Any infinite-loop or skipped-page risk if the
   reaper deletes docs while paginating?
4. Payload layout assumptions in _find_text: what happens with a legit
   long-term memory doc that happens to CONTAIN the substring '"stmm"' in a
   payload value? Could the reaper misparse and then what?
5. K8s manifest: ConfigMap/CronJob consistency (mount path vs command),
   secretRef name, concurrencyPolicy given a sweep that could exceed 15 min,
   image pinned by tag only — what would you change before production?
6. API usage vs src/multimodal_rag/api_server.py: POST body shape (array of
   strings), X-API-Key / X-Dataset-Password header names — confirm against
   the server's auth middleware, not just the route signatures.

## Known gaps (do not re-report)
- No live round-trip test yet (no python interpreter available in the
  authoring session); syntax was reviewed but not executed.
- The reaper trusts created_at from the envelope (client clock), not server
  ingest time. Accepted for v0.1; flag only if you see a worse consequence
  than "an agent with a bad clock shortens/lengthens its own entries' TTL".
