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
const DEFAULT_TOOL_OUTPUT_CHARS = 200
const MAX_TOOL_ARGUMENTS_CHARS = 400
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
  /** Per-tool output preview bound in characters (default 200). */
  toolOutputChars?: number
  /** `source` label stamped on written memories (default "dsh:memory"). */
  source?: string
  /** Set true to disable. */
  disabled?: boolean
}

// ---- text extraction ---------------------------------------------------------

function blockText(block: unknown): string {
  if (!block || typeof block !== 'object') return ''
  const b = block as { type?: string; text?: unknown }
  if (b.type === 'text' && typeof b.text === 'string') return b.text
  return ''
}

function messageText(message: { content?: unknown } | null | undefined): string {
  if (!message || !Array.isArray(message.content)) return ''
  return (message.content as unknown[]).map(blockText).filter(Boolean).join('\n').trim()
}

/** One model-requested tool call, kept so its `tool/result` can name the call and its arguments. */
interface RecordedToolCall {
  name: string
  /** Raw JSON argument string exactly as the model produced it. */
  arguments: string
}

/** Record the tool calls one assistant message requested, keyed by the call id paired with each `tool/result`. */
function recordToolCalls(message: { content?: unknown } | null | undefined, calls: Map<string, RecordedToolCall>): void {
  if (!message || !Array.isArray(message.content)) return
  for (const part of message.content as unknown[]) {
    if (typeof part !== 'object' || part === null) continue
    const b = part as { type?: unknown; id?: unknown; name?: unknown; arguments?: unknown }
    if (b.type !== 'tool-call') continue
    if (typeof b.id !== 'string' || typeof b.name !== 'string') continue
    calls.set(b.id, { name: b.name, arguments: typeof b.arguments === 'string' ? b.arguments : '' })
  }
}

/** Bounded plain-text payload of one tool-result block. */
function toolResultText(block: { content?: unknown } | undefined, bound: number): string {
  if (!block || !Array.isArray(block.content)) return ''
  const texts: string[] = []
  for (const part of block.content as unknown[]) {
    if (typeof part !== 'object' || part === null) continue
    const b = part as { type?: unknown; text?: unknown }
    if (b.type === 'text' && typeof b.text === 'string') texts.push(b.text)
  }
  return texts.join('\n').slice(0, bound)
}

/**
 * Render one `tool/result` event as a Tool section: the requested call with its
 * raw arguments when the pairing assistant message is on the surface, then the
 * preview-bounded result text. A result with neither a pairing call nor
 * non-empty text renders nothing — an empty result carries no recall value.
 */
function toolResultSection(data: unknown, calls: ReadonlyMap<string, RecordedToolCall>, outputBound: number): string {
  const d = (data ?? {}) as { message?: unknown; error?: { code?: unknown } | null }
  const message = d.message as { content?: unknown } | undefined
  const result = Array.isArray(message?.content)
    ? (message.content as unknown[]).find(part => (part as { type?: unknown } | null)?.type === 'tool-result')
    : undefined
  const block = result as { toolCallId?: unknown; content?: unknown; isError?: unknown } | undefined
  const callId = typeof block?.toolCallId === 'string' ? block.toolCallId : ''
  const call = callId === '' ? undefined : calls.get(callId)
  const failure = d.error ?? undefined
  const code = typeof failure?.code === 'string' ? failure.code : ''
  const failed = block?.isError === true || failure !== undefined
  const parts: string[] = []
  if (call) parts.push(`${call.name}(${call.arguments.slice(0, MAX_TOOL_ARGUMENTS_CHARS)})`)
  const output = toolResultText(block, outputBound)
  if (output !== '') parts.push(output)
  if (parts.length === 0) return ''
  const name = call ? call.name : 'unknown tool'
  const heading = `### Tool — ${name}${failed ? ` (error${code === '' ? '' : `: ${code}`})` : ''}`
  return `${heading}\n\n${parts.join('\n\n')}`
}

// ---- transcript builder -------------------------------------------------------

/** Per-call rendering bounds for the memory document. */
export interface HistoryBounds {
  /** Per-tool output preview bound in characters (default 200). */
  toolOutputChars?: number
}

/**
 * Render a session's current surface as the `session_history` memory document.
 *
 * `### User` / `### Assistant` sections carry the messages' plain text. Each
 * `tool/result` becomes a `### Tool — <name>` section holding the call line
 * `name(rawArguments)` followed by the preview-bounded result text; the call
 * comes from the `tool-call` block of the assistant message that requested it,
 * correlated by call id. Failed results add `(error)` or `(error: <code>)` to
 * the heading. A result with neither a pairing call nor non-empty text is
 * omitted. A result whose call block was compacted off the surface renders as
 * `unknown tool` with its result text only.
 *
 * @param surface - current model surface from `sessionQuery.readSurface`.
 * @param bounds - rendering bounds; see {@link HistoryBounds}.
 * @returns the markdown document, with body content truncated at {@link MAX_HISTORY_CHARS}.
 */
export function buildSessionHistory(surface: SessionSurfaceSnapshot, bounds: HistoryBounds = {}): string {
  const outputBound = bounds.toolOutputChars ?? DEFAULT_TOOL_OUTPUT_CHARS
  const events = surface.events
  const sessionId = String(surface.session.id)
  const title = surface.session.cwd ?? surface.session.agentPreset ?? `Session ${sessionId.slice(0, 8)}`
  const started = surface.session.createdAt ? String(surface.session.createdAt) : ''
  const lines: string[] = [`# Session History — ${title}`, '', `- **session:** ${sessionId}`]
  if (started) lines.push(`- **started:** ${started}`)
  let total = lines.join('\n').length
  /** Tool calls requested by assistant messages so far, keyed by call id. */
  const calls = new Map<string, RecordedToolCall>()

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
        recordToolCalls(msg as { content?: unknown }, calls)
        const text = messageText(msg as { content?: unknown })
        if (text) block = `### Assistant\n\n${text}`
        break
      }
      case 'tool/result': {
        block = toolResultSection(data, calls, outputBound)
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
  const toolOutputChars = config.toolOutputChars ?? DEFAULT_TOOL_OUTPUT_CHARS
  if (!Number.isInteger(toolOutputChars) || toolOutputChars < 0) {
    throw new Error(`session-memory-logger: toolOutputChars must be a non-negative integer, got ${String(config.toolOutputChars)}`)
  }

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
        const t = setTimeout(() => { ctl.abort() }, ms)
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
        const doc = buildSessionHistory(surface, { toolOutputChars })
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
    timers.set(sessionId, setTimeout(() => { void flush(sessionId) }, debounceMs))
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

  ctx.on('session/disposed', session => void flush(session.id), { global: true })

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
