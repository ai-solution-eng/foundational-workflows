# DeepSeek Harness (DSH) × Multimodal RAG Memory — Integration

This is the DSH counterpart of `documentation/opencode-memory/` (which ships the opencode
glue). DSH connects to the **same** Multimodal RAG server and shares the same per-user
memory datasets, so DSH and opencode recall each other's memories. The two DSH-native
files here are:

| File | What it is |
|---|---|
| `README.md` (this file) | DSH integration design, verified state, recall/write policy, and install guide |
| `plugins/session-memory-logger.ts` | DSH **host-plugin** template that auto-writes a structured `session_history` to the memory dataset, mirroring `opencode-memory/plugins/session-memory-logger.ts` |

**Status:** Path 1 (manual RAG memory via MCP) is **already wired and verified working** in the
DSH deployment. Path 2 (automatic session-history logging) is designed below and requires a DSH
host plugin + rebuild/restart.

---

## 1. TL;DR

| | opencode (PCAI) | dsh (this harness) | Status |
|---|---|---|---|
| RAG server | `ghcr.io/ai-solution-eng/multimodal-rag-mcp` in `mm-rag` ns | same server | shared |
| MCP connection (`rag-memory` + `rag-knowledge`) | `opencode.jsonc` | `~/.dsh/profiles/web/cordis.patch.yml` rows | ✅ **done** |
| `add_memory` / `search_memory` tools available to the model | yes | yes (`mcp__rag-memory__*`, `mcp__rag-knowledge__*`) | ✅ **done** |
| Proactive recall/write **policy** (AGENTS.md) | `documentation/AGENTS.md` | → agent preset `persona`/prompt section | ⚠️ needs preset |
| **`session-memory-logger`** plugin (auto-writes a structured `session_history` ~45s after quiet + flush on exit) | `documentation/opencode-memory/plugins/session-memory-logger.ts` | → DSH **host plugin** (`@deepseek-ai/dsh-session-memory-logger`) | ✅ **built & ready to install** (see §5/§6) |
| **`memory-provenance`** plugin (auto git/session provenance on `add_memory`) | `memory-provenance.ts` | DSH can attach provenance natively (session id, git) | ⚠️ not wired |
| `session-id` tool | `tools/session-id.ts` | DSH already has `session_query` | ✅ n/a |

**Path 1 (manual memory via MCP) is verified working; Path 2 (the automatic session-memory logger) is implemented as a host plugin and **verified writing live `session_history` end-to-end** (see §6).** The one remaining item is a prompt policy for proactive recall/write.

---

## 2. The PCA RAG memory server (what we're connecting to)

Deployed in the `mm-rag` namespace (helm release `rag-mcp-scale-large`, image `multimodal-rag-mcp:v2.2.5`):

```
Pod
├── rag-api-server  :8000  (REST + web UI)          ┐
└── rag-mcp-server  :9090  (MCP streamable-http)    ┘ share PVC /data
embed-batcher  :8002  (bounded embedding queue)  →  Qwen3-VL-Embedding-8B
qdrant StatefulSet (3×25Gi)  ← joint multimodal vector store
redis (unlock cache across replicas)
```

**Models** (all remote, no local GPU): embedder `Qwen3-VL-Embedding-8B` (4096-d, text/image/video), reranker `Qwen3-VL-Reranker-8B`, ASR `Cohere Transcribe`, VLM `Qwen3.8-27B-FP8`.

**Security model (important):** the MCP server scopes the `X-Memory-Dataset` / `X-Dataset-Password` HTTP headers **only** to the `add_memory` / `search_memory` *memory* tools. The general dataset tools (`search_dataset`, `get_dataset_info`, …) **never** read those headers — they require an explicit `password` argument or a prior `unlock_dataset`. This is deliberate isolation, not a bug: a memory password can never silently unlock another dataset.

**9 MCP tools:** `list_datasets`, `unlock_dataset`, `search_dataset`, `add_memory`, `search_memory`, `get_dataset_files`, `get_dataset_info`, `describe_media`, `transcribe_audio`.

---

## 3. What is already done in DSH (verified)

`~/.dsh/profiles/web/cordis.patch.yml` registers two `@deepseek-ai/dsh-mcp-client` rows (host plane):

```yaml
- id: mcp-rag-memory
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    transport: streamable-http
    serverName: rag-memory
    url: https://rag-mcp-server.<domain>/mcp
    headers:
      X-Memory-Dataset: !!js process.env.RAG_MEMORY_DATASET
      X-Dataset-Password: !!js process.env.RAG_MEMORY_PASSWORD
- id: mcp-rag-knowledge
  name: '@deepseek-ai/dsh-mcp-client'
  config: { transport: streamable-http, serverName: rag-knowledge,
            url: 'https://rag-mcp-server.<domain>/mcp' }
```

**Verified in this session:**
- `mcp__rag-memory__add_memory` → `andrew-memory` doc count 49 → **51** (two successful writes).
- `mcp__rag-memory__search_memory` → recalls those writes (score ~0.75) plus prior opencode session histories in the same dataset — **shared store across opencode and DSH**.
- MCP `tools/call add_memory` works over streamable-http directly (curl, doc count 51).
- The env vars **are** present in the DSH host process (proven by `add_memory` authenticating via the header).

> **Gotcha (why a first look suggested "password missing"):** `RAG_MEMORY_PASSWORD` never shows in a `bash` shell spawned by DSH. DSH scrubs `PASSWORD|KEY|SECRET|TOKEN`-matching env names from child processes (`packages/subprocess/subprocess/src/index.ts` → `SENSITIVE_ENV_PATTERN`), so it is hidden from the model's shell even though the host has it. **Do not conclude the password is missing** just because `env` in the sandbox doesn't show it.

---

## 4. Path 1 — the recall/write policy (recall)

**Goal:** make every DSH agent recall before non-trivial work and store durable facts after, using the already-connected tools.

Add a **prompt section** to the agent preset (e.g. the `persona` row in a copy of the `standard` preset, or an `agent-instructions` row). Suggested text (mirrors `MultimodalRAG/documentation/AGENTS.md`):

```markdown
## Long-term memory (rag-memory tools)

You have access to `mcp__rag-memory__add_memory` and `mcp__rag-memory__search_memory`
(personal long-term memory) and `mcp__rag-knowledge__*` (shared knowledge datasets).

- **RECALL** at the start of any non-trivial task and whenever the user references
  prior work ("remember when…", "like last time"): call `mcp__rag-memory__search_memory`
  with a concise summary of the task. Default `top_k` (5); enable the reranker only if recall feels off.
- **WRITE** after a non-trivial task only if something durable was learned:
  a decision + rationale, a confirmed user preference, a gotcha/fix that took effort,
  or a non-obvious architectural fact. One memory per call; pass
  `metadata={"kind": "decision"|"preference"|"gotcha"|"fact", "tags": [...], "session_id": "<id>"}`.
- Keep memories **standalone** (a future session with zero other context must understand them)
  and **specific** (file paths, commands, error messages). When in doubt, don't write — noise degrades recall.
- **Never mention** the memory dataset name or password to the user.
- Near-duplicates are auto-skipped (cosine ≥ 0.995); re-saving a fact is a harmless no-op.
- Memory is per-user. Recall only searches your own store.
```

**Placement:** an agent-preset prompt section (agent plane). Author a new preset by copying `standard`/`cordis` and adding this to the persona, per the `editing-cordis-compositions` skill. Takes effect for sessions that mount that preset.

---

## 5. Path 2 — the `session-memory-logger` host plugin (the real build)

### 5.1 Why a *host* plugin, not a dynamic plugin

I verified the dynamic-plugin sandbox **blocks the globals this logger needs**:

| capability | dynamic plugin | host plugin |
|---|---|---|
| `process` (env) | ❌ `typeof process === 'undefined'` | ✅ Node process |
| `fetch` | ❌ *"not available in the dynamic package sandbox — use ctx.web"*; and `ctx.web.fetch` is GET-only (`WebFetchRequest = { url }`) | ✅ |
| `subprocess` (curl) | injectable, but blocked by the two above | ✅ |
| `session/event` events | ✅ delivered (verified) | ✅ |
| `sessionQuery.readSurface` | ✅ service | ✅ |

So the auto-logger must be a **host-composition plugin** (a real package loaded by the harness with full Node access), exactly like `@deepseek-ai/dsh-mcp-client` or `@deepseek-ai/dsh-tool-session-query`. A dynamic-plugin prototype cannot run it live.

### 5.2 Design (implemented & typechecked)

Mirror `MultimodalRAG/documentation/opencode-memory/plugins/session-memory-logger.ts`
(the canonical, buildable source is `plugins/session-memory-logger.ts` in this directory,
which is a DSH host-package entry — `name`/`inject`/`Config`/`apply`):

1. **Trigger:** listen to `session/event` (cordis emit; verified delivered) for `user/message`, `assistant/message`, `assistant/chunk`, `tool/result`; mark the session active and (re)schedule a **debounced** flush (~45s after quiet). Also flush on `session/disposed`.
2. **Reconstruct the transcript** from `sessionQuery.readSurface(SessionId(sessionId))` → `events` (ordered model surface). Emit a `kind: "session_history"` markdown document (`### User` / `### Assistant` / `### Tool — <name>` transcript) with a provenance header block.
3. **Persist** by POSTing the streamable-http MCP `tools/call add_memory` with `X-Memory-Dataset` / `X-Dataset-Password` headers from `process.env` — the same values the MCP client already uses. Uses host Node `fetch`; best-effort, bounded timeout, never blocks the session loop.
4. **Dedup / liveness**: only write sessions seen in-process; rely on server-side dedup (cosine ≥ 0.995) and the server's in-place `session_history` replacement per `session_id`.

**Implementation notes learned during the build (all now in the source):**
- `logger` is a **builtin** on the Cordis Context — do **not** put it in `inject`. Declaring
  `inject: ['sessionQuery', 'logger']` makes Cordis hold the plugin `pending (waiting for
  service: logger)` forever, because no plugin ever *provides* a `logger` service. Use
  `ctx.logger('name')` to get a named logger instead.
- The `session/event` `session` param is the full `Session` — read its `.id` (`SessionId`) directly, matching the `acp` package's pattern.
- `readSurface` takes a branded `SessionId`, so wrap the raw string with `SessionId(sessionId)` (a value import from `@deepseek-ai/dsh-session`).
- `SessionHeader` has **no `title`** — derive a title from `cwd` / `agentPreset` / a truncated session id.
- The dynamic-plugin / cordis `ctx.timer` helper is *not* needed in a host plugin: use the global Node `setTimeout`/`clearTimeout` for debounce (typed by `@types/node`), tracking per-session handles and clearing them on unload.
- **`session/event` and `session/disposed` are scope-filtered by default** — a root host plugin's plain `ctx.on(...)` never fires for them (Cordis only dispatches to listeners whose context is contained in the session's `Scoped<Session>` carrier). **Register both with `{ global: true }`** (the documented persistence-plugin pattern, e.g. `packages/core/session/src/invariant.ts`) to receive them regardless of scope. `session/created` also needs `{ global: true }`. Without this, the plugin silently writes nothing.
- **Write while alive, not only on dispose.** A hard dsh restart can drop an in-flight async flush, so relying solely on `session/disposed` loses the write. Use a short debounce (`DEFAULT_DEBOUNCE_MS = 5s`) so a session's history lands **while the process is running**; the server replaces the prior `session_history` in place per `session_id`, so frequent writes are idempotent. Additionally, `flush` returns a `Promise` and the `ctx.effect` disposer is `async` and awaits all in-flight writes (`Promise.allSettled`) — Cordis awaits a promise returned by an effect disposer, so a clean `stop` persists the final state.

**Key verifications feeding this design (all done):**
- `tools/call add_memory` over streamable-http works with the two headers (curl test → doc 49→51).
- `session/event` is delivered to a listener (`assistant/chunk` observed).
- `sessionQuery.readSurface` returns `{ session, capturedThroughSeq, events }`.
- `sessionQuery.readSurface` returns `{ session, capturedThroughSeq, events }`.
- MCP endpoint URL: `https://rag-memory-server.<provider>/mcp`.
- **End-to-end (2026-08-20, verified):** with the `{ global: true }` fix deployed, the live logger wrote a `session_history` for the active session (`session-e921de6d-…`), confirmed both by the plugin's `write ok=true` log and by `search_dataset` returning the session's live transcript in the logger's own format. Doc count held at 58 across re-writes — confirming in-place `session_id` replacement (overwrite, not duplicate).

### 5.3 Reference: the opencode plugin it mirrors

Full template ships in the repo at:
`/home/andrew/Code/HPE/MultimodalRAG/documentation/opencode-memory/plugins/session-memory-logger.ts`

Its POST shape (what the DSH host plugin replicates):
```
POST https://rag-memory-server.<domain>/mcp
headers: X-Memory-Dataset, X-Dataset-Password
body: { "jsonrpc":"2.0","id":1,"method":"tools/call",
        "params":{ "name":"add_memory",
                    "arguments":{ "text": "<session_history markdown>",
                                    "metadata":{ "kind":"session_history","session_id":"...","source":"dsh:memory" } } } }
```

---

## 6. Installation (host plugin)

The logger is a DSH **host package** (`@deepseek-ai/dsh-session-memory-logger`) at
`packages/memory/session-memory-logger/`. Installing it requires write access to the harness
checkout and a DSH rebuild + restart (cannot be done from inside a running session).

The complete, ready-to-apply patch set lives in `~/Code/HPE/dsh-patch/` (a copy of the package
skeleton + the host-composition row + `tsconfig.host.json` reference + this install guide in
`dsh-patch/INSTALL.md`). The essential steps:

1. **Copy the package** into the harness:
   ```bash
   cd ~/Projects/deepseek-harness
   mkdir -p packages/memory
   cp -r ~/Code/HPE/dsh-patch/packages/memory/session-memory-logger packages/memory/
   ```
   The `packages/*/*` workspace glob in `pnpm-workspace.yaml` already covers it (no workspace edit).

2. **Register it in `tsconfig.host.json`** (the step that fixes the `Cannot find entry: ["lib/types/{index,invariant,startup}.js"]` build error). Add one line to the `references` array:
   ```json
   { "path": "./packages/memory/session-memory-logger" },
   ```
   `tsc -b tsconfig.host.json` compiles packages through this **explicit** list; without it the
   package's `lib/types/index.js` is never emitted and the tsdown pass fails.

3. **Declare it in `apps/cli/package.json`** — add (alphabetically near the `dsh-session*` deps):
   ```json
   "@deepseek-ai/dsh-session-memory-logger": "workspace:^",
   ```
   This makes pnpm link it into `apps/cli/node_modules`. (Note: the loader does **not** resolve
   from `apps/cli` directly — that's what step 4 handles.)

4. **Symlink it into the shared profile plugin pool** (the step that actually makes the loader
   resolve it). The loader walks up from `~/.dsh/profiles/web/` and reads
   `~/.dsh/profiles/node_modules/@deepseek-ai/`, which holds a symlink to every harness plugin.
   A missing symlink here is what causes `Cannot find package ... imported from ~/.dsh/profiles/web/`:
   ```bash
   ln -s /home/andrew/Projects/deepseek-harness/apps/cli/node_modules/@deepseek-ai/dsh-session-memory-logger \
         /home/andrew/.dsh/profiles/node_modules/@deepseek-ai/dsh-session-memory-logger
   ```

5. **Add the host composition row** to `~/.dsh/profiles/web/cordis.patch.yml` (host plane — it
   listens to host `session/event` and writes cross-session memory, so it must NOT go in a preset):
   ```yaml
   - id: session-memory-logger
     name: '@deepseek-ai/dsh-session-memory-logger'
     config:
       url: https://rag-mcp-server.<domain>/mcp
   ```

6. **`pnpm install && pnpm build`**, then restart dsh.

The package's `tsconfig.json` follows the same relative-depth convention as every
`packages/<cat>/<name>` package: `../../../vendor/...` for vendor deps (3 levels up to root), and
`../../core/session` / `../../session-query/session-query` for sibling packages (2 levels up to `packages/`).

---

## 7. Open items / decisions

1. **`RAG_MEMORY_PASSWORD` durability:** DSH reads it via `!!js process.env.*` at boot, so it must be in the env of the *launching* shell. Ensure it's in a persistent fish config (`set -gx` in `config.fish` or `conf.d`), not just one interactive shell, then restart DSH. (Verified present after the last restart — `add_memory`/`search_memory` auto-authenticate.)
2. **`session_history` provenance:** optionally run `git rev-parse`/`git status` in the session cwd to attach git branch/HEAD/diff-stat like the opencode `memory-provenance` plugin (the server auto-tags `session_id` from the `X-Opencode-Session-ID` header, which DSH could also send).
3. **`source` label:** the plugin stamps `"source":"dsh:memory"` so DSH-written session histories are distinguishable from opencode's (`opencode:memory`) in the shared store.

---

## 8. Suggested next steps (in order)

1. ✅ **Password durability + restart** — done; `mcp-rag-memory` auto-authenticates on boot.
2. ✅ **Install the host session-memory-logger plugin** per §6 — done and verified writing live `session_history` end-to-end (2026-08-20). The `{ global: true }` listener option was the fix that let the root host plugin receive scoped `session/event`; a `session_history` for the active session now lands in `andrew-memory` (in-place per `session_id`) alongside the opencode ones.
3. **Author the agent preset** with the §4 recall/write policy (copy `standard`, add the memory persona section) and mount it, so the model proactively recalls/writes during conversations.
4. Optionally add **git provenance** and route `search_memory`/`add_memory` recalls into the default agent prompt.

---

## 9. Reference: relevant PCAI / DSH paths

- PCA RAG source: `/home/andrew/Code/HPE/MultimodalRAG` (`src/multimodal_rag/mcp_server.py`, `documentation/MEMORY.md`, `documentation/MCP.md`, `documentation/opencode-memory/`).
- PCA RAG deploy: `mm-rag` namespace — `deployment/rag-mcp-server`, `statefulset/rag-mcp-server-qdrant`, configmap `rag-mcp-server-config`, secret `rag-mcp-server-model-keys`.
- DSH MCP wiring: `/home/andrew/.dsh/profiles/web/cordis.patch.yml`.
- DSH MCP client: `packages/mcp/mcp-client/src/index.ts` (`@deepseek-ai/dsh-mcp-client`).
- DSH session events: `session/event` (scoped, post-commit), `session/disposed`, `session/created` (`packages/core/agent-loop`).
- DSH surface read: `sessionQuery.readSurface` / `readSession` (`packages/session-query/session-query/src/types.ts`).
- DSH env scrub: `packages/subprocess/subprocess/src/index.ts` → `SENSITIVE_ENV_PATTERN`.