# Themes & Working Agreements — the through-lines of our collaboration

**Status:** Compiled 2026-10-02 · Companion to `andrew-card.md` (who) and the per-cluster summaries (what). This card captures the RECURRING patterns — how we work, what we optimize for, and the meta-decisions that repeat across projects.

## Theme 1 — Security-first hardening, pragmatically applied
Every project gets audited (usually by parallel subagents), every audit produces fixes, every fix gets adversarially verified before it ships.
- The identity-ladder pattern (RAG D15–D25) replicated to SQLhandler (identity ladder, policy-as-code, mint_key) and the fleet (mcp_auth, D10 caller attribution).
- SSRF guards, fail-closed defaults, constant-time compares, secrets-shown-once — these are house style now, not one-offs.
- Operator overrides are respected and recorded: keys are private (hide, don't rotate); the browser edge stays open; netzone was rolled back and the doctrine kept.
- Standing rule: cross-validate big changes — a second model reviews the first's work (two waves in the 2026-10-02 audit alone caught 19 defects in already-green code).

## Theme 2 — LLM serving as rigorous systems engineering
GPU serving is treated as measurable arithmetic, not vibes: mamba-slot concurrency math (94→96 via mem-fraction 0.865), DSv4.1's boot-budget-reserve formula, the graph-memory super-linear scaling rule, "ready-state free is a snapshot, not a floor — keep ≥5 GB."
- Every image is digest-pinned; every boot ledger is recorded; every OOM is root-caused (negative mem-usage lines = void the datapoint).
- Benchmarks drive decisions (5.2-vs-5.3 tallies, TTFT recalibration, the SPEC=OFF rule) and the benchmarker itself got fixed when its model was wrong (the KV undercount Andrew challenged).
- Hardware ops have playbooks: drain-check mandates (3 incidents), stuck-PID/XID escalation ladders, teardown-race diagnosis.

## Theme 3 — The agent fleet as infrastructure
"Many small focused servers > one mega-server" (the Sept 8 brainstorm doctrine). The fleet is now: k8s-mcp (read+gated exec), applygate (governed mutation, D11 plan-binding, hash-chained audit), logsearch, workbench (D18 no-python), prometheus, searxng (browser sidecar), sqlhandler, rag, gateway (one front door).
- Fleet conventions: one-address `MCP_API_KEYS` wiring, hardlink mesh via pcai_utils, per-namespace RBAC trios, PCAI paste-values model, values-fulling with `# SITE:` marks.
- The ops loop: "prometheus says what, logsearch+k8s say where/why, applygate fixes."
- Agent teams are the unit of work: parallel deep-divers for audits, zone-partitioned implementers for fixes, adversarial verifiers for trust.

## Theme 4 — MultimodalRAG as the flagship product
Built, audited, hardened, and released weekly since August: 17+ format ingestion, caption twins, hybrid RRF retrieval, reranker, media-lite fetch, D23 four-tab SPA, D15–D25 identity ladder, federated search, the whole 974-test suite.
- Its history also encodes the working relationship: every feature (twins, RRF, D22 SSO, D25 self-mint) came from a session where Andrew described the outcome he wanted and the agents delivered in agent-team mode.
- The 2026-10-02 arc (audit → 2×P0 → remediation → 2 cross-validation waves → perf wave → v5.4.1 same-day release) is the template for how big changes should run.

## Theme 5 — Memory systems as first-class infrastructure
The reason this dataset exists. Andrew treats recall as a product: the bloat postmortems (Sept 9, and the 2026-10-02 rebuild), the francesco-memory migration (bm25 backfill, point-copy schema upgrade), the DSH auto-flush ambition ("automatic, or basically a single button press summarize + ship").
- Design principle settled 2026-10-02: the memory dataset is a PROJECT MAP (what we worked on / how to access it / how to proceed), structured as distilled per-project summaries + atomic fact cards + a WHO/themes card — NOT raw transcripts. Transcripts stay local (DSH: ~/.dsh/sessions/) or in backup exports.
- Chunking: 8192 tokens default (dataset splits automatically).

## Standing working agreements (do not re-litigate)
1. Never run git state operations (checkout/restore/clean/stash) in Andrew's repos — uncommitted work is normal; ask first.
2. Mask secrets in all reports (`ey***fQ`); never print full key material.
3. PCAI deploys are values-paste: local values files are FULL standalone docs with `# SITE:` marks.
4. Adversarial cross-validation on significant changes; implementer prompts must forbid git state ops and timeout-wrap test runs (the bwrap orphaned-process lesson).
5. Back up datasets before destructive migrations (the pre-Aug-19 memory loss).
6. Hardlink mesh: audit (inode-level) before relinking; Syncthing silently re-inodes mirror files; frozen-vintage snapshots are deliberate — don't force-link.
7. Fish shell, `G` function for kubectl+G2, socks5 proxy for lab traffic from the laptop.
8. When Andrew challenges a technical claim, re-derive from first principles — he's usually spotted something real (three documented cases: mamba direction, DSv4.1 KV estimate, the netzone browser dependency).
