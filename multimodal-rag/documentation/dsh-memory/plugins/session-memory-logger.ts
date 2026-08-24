/**
 * DSH session-memory-logger — host plugin
 *
 * After a DSH session goes quiet (~45s) or is disposed, reconstruct the session's
 * ordered model surface from `sessionQuery.readSurface` and write it to the shared
 * Multimodal RAG memory dataset as a `kind: "session_history"` memory, so future
 * DSH/opencode sessions can recall it with `search_memory`.
 *
 * Mirrors MultimodalRAG/documentation/opencode-memory/plugins/session-memory-logger.ts.
 * Design + verification notes: MultimodalRAG/documentation/dsh-memory/README.md
 *
 * HOST-PLANE plugin (not a dynamic plugin): it needs the full Node globals
 * `process` (env) and `fetch` (network), which the dynamic-package sandbox blocks.
 *
 * Runtime requirements:
 *   - `RAG_MEMORY_DATASET` and `RAG_MEMORY_PASSWORD` in the dsh process env
 *     (same vars the `mcp-rag-memory` client row already reads).
 *   - services `sessionQuery`, `timer`, `logger`.
 *
 * @module @deepseek-ai/dsh-session-memory-logger
 */

import type { Context } from '@deepseek-ai/cordis'
import type { SessionEvent } from '@deepseek-ai/dsh-session'
import { SessionId } from '@deepseek-ai/dsh-session'
import type { SessionSurfaceSnapshot } from '@deepseek-ai/dsh-session-query'

/** Cordis plugin name used by Loader diagnostics. */
export const name = 'session-memory-logger'

/** Capability services required by this plugin. `logger` is intentionally NOT injected:
 * it is a builtin on the Cordis Context (`ctx.logger(name)`), so declaring it here would
 * make Cordis hold this plugin `pending` waiting for a `logger` service that is never
 * provided by any plugin. Use `ctx.logger(name)` directly instead. */
export const inject = ['sessionQuery']

// Debounce before writing a session's history. The server replaces the prior
// `session_history` in place per `session_id` (cosine dedup + in-place swap), so
// frequent writes are idempotent. Keep it short so a session is persisted WHILE
// the dsh process is alive — a hard restart can drop an in-flight async flush,
// so the write must land before teardown rather than only on dispose.
const DEFAULT_DEBOUNCE_MS = 5_000
const POST_TIMEOUT_MS = 20_000
const MAX_TOOL_OUTPUT_CHARS = 2000
const MAX_HISTORY_CHARS = 32_000

interface MemoryServer {
  url: string
  dataset: string
  password: string
  source: string
}

/** Minimal fetch-like surface so the transport can be swapped for subprocess+curl. */
interface HttpTransport {
  post(url: string, headers: Record<string, string>, body: string, timeoutMs: number): Promise<{ ok: boolean; body: string }>
}

/** Deployment config. */
export interface Config {
  /** Streamable-HTTP MCP endpoint of the Multimodal RAG server, e.g. https://rag-memory-server.<domain>/mcp */
  url: string
  /** Env var name holding the memory dataset name (default RAG_MEMORY_DATASET). */
  datasetEnv?: string
  /** Env var name holding the dataset password (default RAG_MEMORY_PASSWORD). */
  passwordEnv?: string
  /** Quiet-window before writing, in ms (default 45000). */
  debounceMs?: number
  /** `source` label stamped on written memories (default "dsh:memory"). */
  source?: string
  /** Set true to disable. */
  disabled?: boolean
}

// ---- text extraction ---------------------------------------------------------

function blockText(block: unknown): string {
  if (!block || typeof block !== 'object') return ''
  const b = block as { type?: string; text?: unknown; name?: unknown; content?: unknown }
  if (b.type === 'text' && typeof b.text === 'string') return b.text
  if (b.type === 'tool_use' && typeof b.name === 'string') return `[tool: ${b.name}]`
  if (b.type === 'tool_result' && b.content) {
    const c = Array.isArray(b.content)
      ? (b.content as { text?: string }[]).map((x) => (x && typeof x.text === 'string' ? x.text : '')).join(' ')
      : String(b.content)
    return c.slice(0, MAX_TOOL_OUTPUT_CHARS)
  }
  return ''
}

function messageText(message: { content?: unknown } | null | undefined): string {
  if (!message || !Array.isArray(message.content)) return ''
  return (message.content as unknown[]).map(blockText).filter(Boolean).join('\n').trim()
}

// ---- transcript builder -------------------------------------------------------

export function buildSessionHistory(surface: SessionSurfaceSnapshot): string {
  const events = surface.events ?? []
  const sessionId = String(surface.session?.id ?? 'unknown')
  const title = surface.session?.cwd ?? surface.session?.agentPreset ?? `Session ${sessionId.slice(0, 8)}`
  const started = surface.session?.createdAt ? String(surface.session.createdAt) : ''
  const lines: string[] = [`# Session History — ${title}`, '', `- **session:** ${sessionId}`]
  if (started) lines.push(`- **started:** ${started}`)
  let total = lines.join('\n').length

  for (const ev of events) {
    const data = (ev as { data?: unknown }).data
    let block = ''
    switch (ev.type) {
      case 'user/message': {
        const msg = (data && (data as { message?: unknown }).message) ?? data
        const text = messageText(msg as { content?: unknown })
        if (text) block = `### User\n\n${text}`
        break
      }
      case 'assistant/message': {
        const msg = (data && (data as { message?: unknown }).message) ?? data
        const text = messageText(msg as { content?: unknown })
        if (text) block = `### Assistant\n\n${text}`
        break
      }
      case 'tool/result': {
        const d = (data ?? {}) as { tool?: unknown; output?: unknown }
        const tn = d.tool ? String((d.tool as { name?: string }).name ?? d.tool) : 'result'
        const out = typeof d.output === 'string' ? d.output : JSON.stringify(d.output ?? '')
        block = `### Tool — ${tn}\n\n${out.slice(0, MAX_TOOL_OUTPUT_CHARS)}`
        break
      }
      default:
        break
    }
    if (!block) continue
    lines.push(block)
    total += block.length
    if (total > MAX_HISTORY_CHARS) {
      lines.push('\n_[history truncated]_')
      break
    }
  }
  return lines.join('\n\n')
}

// ---- MCP write ----------------------------------------------------------------

export function buildAddMemoryBody(doc: string, sessionId: string, source: string): string {
  return JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'tools/call',
    params: {
      name: 'add_memory',
      arguments: {
        text: doc,
        metadata: { kind: 'session_history', session_id: sessionId, source },
      },
    },
  })
}

export async function postSessionHistory(
  server: MemoryServer,
  doc: string,
  sessionId: string,
  transport: HttpTransport,
): Promise<{ ok: boolean; body: string }> {
  return transport.post(server.url, {
    'Content-Type': 'application/json',
    Accept: 'application/json, text/event-stream',
    'X-Memory-Dataset': server.dataset,
    'X-Dataset-Password': server.password,
  }, buildAddMemoryBody(doc, sessionId, server.source), POST_TIMEOUT_MS)
}

// ---- plugin apply ------------------------------------------------------------------

export function apply(ctx: Context, config: Config): void {
  const log = ctx.logger('session-memory-logger')
  const dataset = process.env[config.datasetEnv ?? 'RAG_MEMORY_DATASET'] ?? ''
  const password = process.env[config.passwordEnv ?? 'RAG_MEMORY_PASSWORD'] ?? ''
  const debounceMs = config.debounceMs ?? DEFAULT_DEBOUNCE_MS

  if (config.disabled || !config.url || !dataset || !password) {
    log.warn('disabled — need url + RAG_MEMORY_DATASET/PASSWORD in the dsh process env')
    return
  }

  const server: MemoryServer = { url: config.url, dataset, password, source: config.source ?? 'dsh:memory' }
  const active = new Set<string>()
  const timers = new Map<string, ReturnType<typeof setTimeout>>()
  // Track in-flight writes so plugin disposal can await them (Cordis awaits a
  // promise returned by an effect disposer), letting a clean stop persist the
  // final state instead of dropping the async POST on process exit.
  const pending = new Set<Promise<void>>()

  // HTTP transport: prefer global fetch (host Node).
  let transport: HttpTransport | null = null
  if (typeof fetch === 'function' && typeof AbortController !== 'undefined') {
    transport = {
      async post(u, headers, body, ms) {
        const ctl = new AbortController()
        const t = setTimeout(() => ctl.abort(), ms)
        try {
          const r = await fetch(u, { method: 'POST', headers, body, signal: ctl.signal })
          return { ok: r.ok, body: await r.text() }
        } finally {
          clearTimeout(t)
        }
      },
    }
  }

  // Returns a promise so callers can await the write (dispose/teardown).
  const flush = (sessionId: string): Promise<void> => {
    if (!active.delete(sessionId)) return Promise.resolve()
    const t = timers.get(sessionId)
    if (t !== undefined) clearTimeout(t)
    timers.delete(sessionId)
    const p = (async () => {
      try {
        const surface = await ctx.sessionQuery.readSurface(SessionId(sessionId))
        const doc = buildSessionHistory(surface)
        if (!doc || doc.length < 20) return
        if (!transport) {
          log.warn('no fetch transport; skipping write')
          return
        }
        const res = await postSessionHistory(server, doc, sessionId, transport)
        log.info(`write ok=${res.ok} ${res.body.slice(0, 120)}`)
      } catch (e) {
        log.warn(`flush error: ${String(e)}`)
      }
    })()
    pending.add(p)
    void p.finally(() => pending.delete(p))
    return p
  }

  const scheduleFlush = (sessionId: string): void => {
    // Debounce: cancel any pending flush for this session, schedule a fresh one.
    const existing = timers.get(sessionId)
    if (existing !== undefined) clearTimeout(existing)
    timers.set(sessionId, setTimeout(() => flush(sessionId), debounceMs))
  }

  // CRITICAL: `session/event` and `session/disposed` are dispatched scoped to the
  // session carrier by default — a root host plugin's plain `ctx.on(...)` would
  // never fire. Opt into `{ global: true }` to receive them regardless of scope
  // (the documented persistence-plugin pattern; see `packages/core/session`).
  ctx.on('session/event', (session, event: SessionEvent) => {
    const t = (event as { type?: string }).type
    if (t !== 'user/message' && t !== 'assistant/message' && t !== 'assistant/chunk' && t !== 'tool/result') return
    const sid = session.id
    active.add(sid)
    scheduleFlush(sid)
  }, { global: true })

  ctx.on('session/disposed', (session) => void flush(session.id), { global: true })

  // Async disposer: Cordis awaits the returned promise, so the final writes for
  // every active session are awaited on a clean stop.
  ctx.effect(() => async () => {
    for (const sid of Array.from(active)) void flush(sid)
    active.clear()
    for (const t of timers.values()) clearTimeout(t)
    timers.clear()
    await Promise.allSettled(Array.from(pending))
  })
}