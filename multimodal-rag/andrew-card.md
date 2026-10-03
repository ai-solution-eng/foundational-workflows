# Andrew — Who You're Working With (the WHO card)

**Status:** Compiled 2026-10-02 from the full memory corpus · **This card exists so any future session starts knowing the person, not just the projects.**

## Identity & environment
- **Andrew Bydlon** — HPE, AI Solution Engineering (SE). Kubernetes identity `andrew-bydlon`; G2 cluster namespace `project-user-andrew-bydlon`; gateway/model identity for keys is `andrew-bydlon`.
- **Main workstation:** `andrewMain` — CachyOS (Arch-based) + Hyprland + ML4W dotfiles, AMD Ryzen (Matisse/X570), NVIDIA RTX 4090. Suspend bug diagnosed+fixed Aug 2026. Second machine for Anna: Ubuntu 26.04 with ml4w port kit.
- **Shell: fish** (has a `G` function wrapping kubectl + the G2 kubeconfig at `~/SSH/G2/g2_kubeconfig.yaml` with the socks5h://127.0.0.1:1080 lab proxy). Prefers `rcat` over cat. Uses Syncthing across machines (folder vs2ta-dvzqu = ~/Code/HPE) — which silently replaces hardlinked mirror files with new inodes; audit before relinking.
- Works from the laptop against the **G2 lab cluster** (`pcai-se-ai-application.hst.rdlabs.hpecorp.net` — PCAI/Ezmeral + EzUA) through ezaf-gateway; svc DNS not resolvable locally, everything rides external gateway URLs with static Bearer keys.

## How Andrew works (pattern, verified across ~90 sessions)
- **Agent-team driven, high throughput.** Almost every session spawns parallel subagents — audits fan out 4-8 deep-divers, implementation is zone-partitioned across implementers, and he explicitly asks for cross-validation ("use agent teams as necessary", "cross validate your work with Deepseek, and vice versa"). He expects the models to check each other.
- **Decisive with strong opinions, and he challenges the model when the model is wrong** — and when challenged, re-derive from first principles rather than defend. Examples: the mamba-budget direction correction (Sept 3), the DSv4.1 KV-estimate challenge (the 11M-token launch memory vs the stale 2.10M artifact), the netzone rollback ("I didn't want to break the browser frontend! Let's undo this change." → doctrine locked same day).
- **Security-conscious but pragmatic.** Ratified blanket-approval for hardening waves (D1–D17), but with explicit operator overrides: "Don't rotate keys — they are private. Just worry about hiding them"; rejected the netzone browser lockdown; rejected danger-full-access escalations; secrets stay private, hide don't rotate. Security posture: browser/edge stays open, machine traffic keyed, admin surfaces gated.
- **Hates waste and noise:** killed a runaway test process mid-session ("fans are spinning... seems a bit ridiculous"); asked for the memory store to be de-bloated twice; wants automatic distillation instead of manual memory work ("If not purely automatic, basically a single button press").
- **Runs everything as agent-accessible infrastructure:** MCP fleet (k8s-ops, applygate, logsearch, workbench, searxng, prometheus, rag, sqlhandler, gateway) — his tools are first-class projects he audits and hardens like products.

## Working agreements established over time
- Never run `git checkout/restore/clean/stash` in his repos without asking — uncommitted work is the norm (he hardlinks and syncs manually).
- PCAI deployments are values-paste (no helm CLI): every `helm/local/values*.yaml` is a FULL standalone paste-ready doc with `# SITE:` marks.
- Secrets are private: never print full key material; mask in reports (`ey***fQ` style).
- DSH sessions: transcripts stay local (`~/.dsh/sessions/`); the memory dataset is for distilled knowledge (see the 2026-10-02 rebuild decision).
- Backups before destructive migrations (the pre-2026-08-19 memory loss lesson — old Qdrant PVCs deleted without backup).
- Adversarial verification is expected on big changes: a second model reviews the first's work before it ships.

## Recurring themes (the through-lines of 2026)
1. **Security-first hardening** — audits of every surface (RAG, SQL, fleet), identity ladders (D15–D25), SSRF guards, mcp-netzone (rolled back but doctrine kept), secret-hiding.
2. **LLM serving on H200s** — GLM-5.3-Flash + DeepSeek-V4.1-Flash on SGLang: memory-fraction arithmetic, speculative decoding (EAGLE/DFlash/DSPARK), HiCache tiers, concurrency math, benchmark-driven tuning.
3. **The agent fleet as a product** — many small focused MCP servers, one gateway front door, governed mutation (applygate D11), per-namespace RBAC, OWUI onboarding.
4. **MultimodalRAG as the flagship** — 17+ formats, hybrid RRF, caption twins, D23 four-tab SPA; audited, hardened, released weekly (v3.x → v5.4.1).
5. **Memory systems as a first-class concern** — andrew-memory, francesco-memory migrations, the bloat postmortems, this 2026-10-02 rebuild; he wants recall of people/projects/decisions, automatic and unbloated.
