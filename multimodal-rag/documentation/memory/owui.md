# Open WebUI — Memory via the Filter

Open WebUI gets per-user long-term memory through the filter in [`openwebui_extension/`](../../openwebui_extension/) (`filter.py`), not through MCP: an **inlet** auto-recalls at conversation start and
an **outlet** distills and stores after each reply — both via the RAG REST API directly. Media routing (staging + MCP tool hints) is the other half of the same filter; the full valve reference lives
in the extension's [README.md](../../openwebui_extension/README.md). Store mechanics shared with the other clients: [README.md](README.md).

## What the filter does

1. **Recall (inlet):** at the start of each conversation (first user message by default, `MEMORY_RECALL_FIRST_ONLY`), the filter searches that user's memory dataset and injects the top-k relevant
   memories as context (`MEMORY_RECALL_TOP_K`, default 5; injected as a system message when `MEMORY_INJECT_AS_SYSTEM`). Recall runs before media processing — a single message can trigger both a memory
   recall and media staging.
2. **Write (outlet):** after the LLM replies, the filter asks a separate **distillation LLM** (`DISTILL_LLM_URL` / `DISTILL_LLM_MODEL` / `DISTILL_LLM_API_KEY` valves) to extract durable facts from the
   exchange; whatever it produces (anything but `NOTHING`) is stored in the user's memory dataset via the REST API. The user sees nothing — no tool calls in chat, no password in context. The
   distillation LLM can be any small fast model; it does not need to be the model the user is chatting with.

The memory-free media-only variant is `filter_media_strip.py` (strips image/video/audio parts for text-only LLMs, no RAG). There is no separate "no memory" fork — disable `MEMORY_ENABLED` /
`SQL_LESSONS_ENABLED` in `filter.py` instead.

## Per-user isolation (SSO-backed)

OWUI filter Valves are **global** (admin-configured once, shared by all users on the instance), so per-user passwords can't live in Valves. The filter derives **two** per-user secrets from the
SSO-authenticated `__user__` identity at runtime:

```
dataset_name = MEMORY_DATASET_PREFIX + sanitised(__user__.id)
             e.g. "owui-memory-a1b2c3d4"

password     = HMAC-SHA256(MEMORY_SECRET, __user__.id)[:18]   (base64url, 24 chars)
             e.g. "c82tY2vCJGCxRjTwr7MDxYxs"
```

Because OWUI populates `__user__` **after** SSO authentication, a user cannot forge another user's id — the derivation is sound.

| | `MEMORY_SECRET` set (recommended) | `MEMORY_SECRET` empty (fallback) |
|---|---|---|
| Dataset name | per-user (from `__user__.id`) | per-user (from `__user__.id`) |
| Dataset password | per-user (HMAC-derived, unpredictable) | shared (`MEMORY_PASSWORD`) |
| If password leaks | one user exposed | all users exposed |
| Admin setup | one `MEMORY_SECRET` string | one `MEMORY_PASSWORD` string |

## Setup (SSO-enabled, recommended path)

1. Generate a random secret: `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
2. Install the filter (Admin Panel → Functions → paste `filter.py`).
3. In the filter ⚙️ set:
   - `MEMORY_DATASET_PREFIX` (default `owui-memory-` is fine)
   - `MEMORY_SECRET` (your random secret)
   - `MEMORY_AUTO_CREATE = true` (zero per-user provisioning — the filter creates each user's dataset with their derived password on first write)
   - `DISTILL_LLM_URL` / `DISTILL_LLM_MODEL` / `DISTILL_LLM_API_KEY`
4. Leave `MEMORY_ENABLED = true`. Recall starts immediately (empty for users whose dataset doesn't exist yet); writes start once the distillation LLM is configured.

Related valves worth knowing: `MEMORY_RECALL_TOP_K`, `MEMORY_RECALL_FIRST_ONLY`, `MEMORY_INJECT_AS_SYSTEM`, `DISTILL_MIN_REPLY_CHARS` (default 200 — shorter replies are not distilled). Full table in
the [extension README](../../openwebui_extension/README.md).

> **Secret rotation:** changing `MEMORY_SECRET` re-derives all per-user passwords, making existing datasets inaccessible. To rotate: re-create each user's dataset, or update each dataset's password via
> the REST API.

## Sharing with opencode / DSH

One dataset per user can be shared across all clients (recommended) — point opencode's `RAG_MEMORY_DATASET` at the same dataset and set its password to the same value, or give OWUI a
`MEMORY_DATASET_PREFIX` that matches a fixed opencode dataset for single-user setups. Memories carry `source: "openwebui:memory"` vs `"opencode:memory"` / `"dsh:memory"` so they're attributable; see
[README.md](README.md) § How memories are written.
