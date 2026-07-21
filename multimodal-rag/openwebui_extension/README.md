# Multimodal RAG Bridge — Open WebUI Extension

A filter function for [Open WebUI](https://docs.openwebui.com/) that hands
unsupported modalities (images, video, audio) to the Multimodal RAG MCP
tool **without** injecting raw media into the LLM context window.

## Problem

Open WebUI blocks requests containing modalities (images, video, audio)
that the target LLM doesn't support. Even when uploads succeed with a
capable model, the raw media data (e.g. base64-encoded images) is injected
into the LLM context window, consuming valuable tokens.

This extension solves that by **deferring to the RAG MCP tool** instead of
embedding media (or preemptively retrieved RAG text) in the context:

1. **Allows uploads** of any modality without requiring the LLM to support
   them natively.
2. **Stages the media** on the RAG API's staging endpoint and injects only
   a short `file://` URL hint plus the dataset name — **no base64** in the
   LLM context.
3. Lets the **LLM call the `search_dataset` MCP tool itself** with that
   URL when it decides retrieval is relevant, so results arrive as tool
   results (not as silently injected context).

If the RAG MCP is not enabled (`DEFER_TO_MCP = false`), the filter simply
warns the user and strips the unsupported modality.

## How It Works

```
User uploads files / pastes image  ──→  Open WebUI Filter (inlet)
                                            │
                              ┌─────────────┴─────────────┐
                              │                           │
                      Files/media found            No files → pass through
                              │
             ┌─────────────────┼──────────────────┐
             │                 │                  │
        Media files        Text files         Binary files
       (img/video/audio)   (.txt/.md/...)     (PDF/docx/...)
             │                 │                  │
             ▼                 ▼                  ▼
   DEFER_TO_MCP?          Read content       Include filename
      │   │               directly           as reference
   true   false                │                  │
      │   └→ warn + strip      │                  │
      │       (no RAG)         │                  │
      ▼                        │                  │
  POST /api/staging            │                  │
  (RAG API server)             │                  │
      │                        │                  │
      ▼                        │                  │
  file:// URL hint             │                  │
  + dataset name               │                  │
      │                        │                  │
      └─────────┬──────────────┘──────────────────┘
                │
                ▼
      Inject text hint + context
      (system or user message)
                │
                ▼
        LLM receives only text
                │
                ▼ (if DEFER_TO_MCP)
   LLM calls search_dataset MCP tool
   with the staged file:// URL
                │
                ▼
   Multimodal RAG API (MCP server reads
   the staged file from the shared PVC)
                │
                ▼
   Retrieval results returned to the LLM
   as a tool result
```

### What happens to different file types

| File type | Behavior |
|-----------|----------|
| **Image** (jpg, png, gif, etc.) | Staged on the RAG API; a `file://` URL hint is injected so the LLM can call `search_dataset` (when `DEFER_TO_MCP=true`). Stripped + warned when `DEFER_TO_MCP=false`. |
| **Video** (mp4, webm, etc.) | Same as image. |
| **Audio** (mp3, wav, flac, etc.) | Same as image. |
| **Text** (txt, md, json, etc.) | Read directly and included as inline context. |
| **Other** (PDF, docx, etc.) | Filename included as a reference (full text extraction requires additional libraries). |

## Prerequisites

- Open WebUI **v0.3.0+** (for Functions/Filter support)
- A running [Multimodal RAG API server](src/multimodal_rag/api_server.py)
  (for the staging endpoint) — ships with the standard helm chart
- The **Multimodal RAG MCP server** enabled and connected to the Open
  WebUI model, so the LLM can call `search_dataset`.
- The API and MCP containers must share the `DATA_PATH` PVC (they do in
  the default helm deployment: both mount `/data`), so the MCP server can
  read staged `file://` URLs directly.

## Installation

1. Open your Open WebUI admin panel.
2. Navigate to **Admin Panel → Functions**.
3. Click **+** (Add Function).
4. Paste the contents of `filter.py` into the editor.
5. Set the **Name** to `Multimodal RAG Bridge`.
6. Click **Save**.

The filter will appear in the Integrations menu (⚙️) and can be toggled
per-chat via the chip icon.

## Configuration

After installing, click the ⚙️ icon next to the filter to configure:

| Valve | Default | Description |
|-------|---------|-------------|
| `RAG_API_URL` | `http://rag-api:8000` | Base URL of the Multimodal RAG API (for the staging endpoint + dataset list) |
| `DATASET_NAME` | `default` | Fallback injected into the hint only if the filter can't fetch the live dataset list |
| `ROUTE_IMAGES` | `true` | Hand images off to the MCP tool (false = leave for a vision LLM) |
| `ROUTE_VIDEO` | `true` | Hand video off to the MCP tool |
| `ROUTE_AUDIO` | `true` | Hand audio off to the MCP tool |
| `STRIP_MODELS` | _(empty)_ | Comma-separated text-only model IDs/names where media is stripped. Models NOT listed keep media in context (vision LLMs). Empty = strip for all. |
| `DEFER_TO_MCP` | `true` | Stage media + inject URL hint so the LLM calls `search_dataset`. **false** = no staging (text-only warns+drops, vision passes through). |
| `STAGING_PATH` | `/api/staging` | Path on the RAG API server used to stage uploaded media |
| `MCP_TOOL_HINT` | _(default text)_ | Header prepended to the staged-media hint |
| `INJECT_AS_SYSTEM` | `true` | Inject as system message (vs. append to user) |
| `MAX_CONTEXT_CHARS` | `4000` | Max characters of injected context (hint + text files) |
| `CONTEXT_HEADER` | _(default text)_ | Header before injected context |

## Model Setup in Open WebUI

For the frontend to **allow** media uploads, the selected model must report
that it supports those modalities. You can configure this in:

**Admin Panel → Models → (your model) → Capabilities**

Set `vision: true` even if the actual LLM doesn't support vision. The filter
will intercept the images before they reach the LLM.

To let the LLM call the RAG MCP tool, attach the **Multimodal RAG MCP
server** to the model under **Admin Panel → Models → (your model) →
Connections / Tools** (the exact location depends on your Open WebUI
version). When MCP is attached, keep `DEFER_TO_MCP = true`.

### Text-only vs vision LLMs (`STRIP_MODELS`)

The filter can serve both text-only and vision LLMs from a single
installation. Set `STRIP_MODELS` to a comma-separated list of your
text-only model IDs or names:

| Model type | In `STRIP_MODELS`? | `DEFER_TO_MCP=true` | `DEFER_TO_MCP=false` |
|------------|-----|---------------------|----------------------|
| Text-only (e.g. Deepseek) | yes | Strip image → stage → hint → LLM calls `search_dataset` | Warn + drop image |
| Vision (e.g. Gemma) | no | **Keep image in context** + stage → hint → LLM sees image natively AND can call `search_dataset` | Image passes through to LLM natively (no RAG) |

**Example:** with `STRIP_MODELS = "deepseek-v4"`:
- Deepseek v4 → image stripped, staged, hint injected (text-only RAG)
- Gemma 31b → image stays in context AND is staged with hint (vision + optional RAG)

Both models can be used in the same Open WebUI instance with the same
filter — the filter checks `__model__` at runtime.

If your model natively supports some modality (e.g. text + images) and you
do **not** want to route those through RAG at all, set the matching
`ROUTE_*=false` to let them pass through unchanged.

## Usage

### With RAG MCP enabled (default)

1. **Toggle the filter on** in the chat (click the chip icon).
2. **Upload media** using the paperclip button or paste images directly.
3. **Ask your question** — the filter stages the media and injects a short
   URL hint (no base64). The LLM calls `search_dataset` with that URL when
   it needs context.

### With RAG MCP disabled (`DEFER_TO_MCP = false`)

1. Upload media as usual.
2. The filter warns you that the modality is unsupported and strips it
   from the request; the LLM only sees the text.

### Example (DEFER_TO_MCP = true)

**User uploads:** `photo.jpg` (a picture of a dog in a park)
**User asks:** "What breed is this dog?"

**With filter:**
1. Image bytes are uploaded to `POST /api/staging` on the RAG API.
2. The filter fetches the live dataset list from `GET /api/datasets`
   (6 datasets in the current deployment, e.g. `andrew-test-dataset`,
   `stacks-project`, ...).
3. The image is stripped from the LLM request; only a hint is injected:
   `<staged image (file:///data/staging/<uuid>/photo.jpg) — available
   datasets: <list>; call search_dataset(image="<url>",
   dataset_name="<chosen>") to retrieve context>`.
4. The LLM picks the most relevant dataset and calls `search_dataset`
   with the staged `file://` URL.
5. The MCP server reads the image from the shared PVC and retrieves
   related context (e.g. _"Image shows a Golden Retriever in a park..."_).
6. The LLM uses that tool result to answer.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Media files not uploaded | Model lacks capability flags | Enable `vision` etc. in model config |
| No MCP tool call happens | MCP server not attached to the model | Attach the Multimodal RAG MCP server in model settings, keep `DEFER_TO_MCP=true` |
| "media skipped — DEFER_TO_MCP off" warning | `DEFER_TO_MCP=false` and no native support | Enable MCP + set `DEFER_TO_MCP=true`, or set `ROUTE_*=false` for a vision LLM |
| Staging upload fails | RAG API unreachable | Check `RAG_API_URL` and `STAGING_PATH` |
| MCP tool can't read `file://` URL | PVC not shared between API & MCP | Ensure both containers mount `DATA_PATH` at the same path (default helm chart does) |
| Staged files accumulate | Sweep runs probabilistically (~1 in 10 uploads) in the background | Increase `STAGING_SWEEP_RATE` (or set `STAGING_TTL` lower); files live under `DATA_PATH/staging/` |
| Hint shows only `DATASET_NAME` instead of full list | Filter couldn't reach `GET /api/datasets` | Check `RAG_API_URL` reachability and RAG API health |
| Upload shows error in UI | `file_handler` conflict with other filters | Check for conflicting filters |

## Architecture

```
┌──────────┐   uploads   ┌──────────────────┐  POST    ┌─────────────────┐
│  Open    │ ──────────→ │  Filter (inlet)  │ ────────→│  Multimodal     │
│  WebUI   │             │                  │ /staging │  RAG API        │
│  Chat    │ ←────────── │  • Strip media   │ ←────────│  (writes to PVC)│
└──────────┘   hint only │  • Read text     │ file://  └─────────────────┘
      │                  │  • Stage media   │               │
      │                  │  • Inject hint   │               ├─ Qdrant (vector store)
      │                  └──────────────────┘               ├─ Embedding Model
      │                       │                             ├─ VLM (vision→text)
      │                       │ tool call                   ├─ ASR (audio→text)
      │                       ▼                             └─ Reranker
   ┌──────────┐         ┌──────────────┐
   │  LLM     │ ──────→ │  RAG MCP     │ ── reads file://──→ (shared PVC)
   │ (text    │ ←────── │  server      │
   │  only)   │ result  │ search_      │
   └──────────┘         │  dataset()   │
                        └──────────────┘
```

The filter uses `file_handler = True` to take full control of file processing,
bypassing Open WebUI's built-in text-only RAG pipeline. This prevents garbage
output from trying to embed images/video/audio as text.
