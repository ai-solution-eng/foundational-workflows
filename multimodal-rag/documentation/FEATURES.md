# Multimodal RAG — Supported Formats & Processing Details

## Overview

This system ingests documents in **17+ file formats**, processes each with a format-specific chunking strategy, embeds them into a joint multimodal vector space (text, image, video, audio) via **Qwen3-VL-Embedding-8B**, and retrieves them at query time with optional cross-encoder reranking via **Qwen3-VL-Reranker-8B**.

Modalities the embedder doesn't support natively (audio) are converted to text via ASR **before** embedding (Preprocessor). Modalities the downstream LLM doesn't support are converted **after** retrieval (Postprocessor).

Chunk sizes are **dynamic** — sourced from the embedder model config, with a dual-budget system: general text uses `chunk_size=2048` / `chunk_overlap=256`, while structured types (code/json/xml/yaml) use `code_chunk_size=8192` / `code_chunk_overlap=512`. When a HuggingFace tokenizer is available, chunking is **token-aware** rather than character-based.

---

## Archive Formats

| Format | Backend |
|--------|---------|
| `.zip` | `zipfile` (stdlib) |
| `.tar`, `.tar.gz`/`.tgz`, `.tar.bz2`/`.tbz2`, `.tar.xz`/`.txz` | `tarfile` (stdlib) |
| `.rar` | `rarfile` or `unrar` CLI fallback |

**Flow:**
1. Extract to a temp directory using the appropriate backend.
2. Walk the extracted tree and dispatch each contained file to its dedicated processor (PDF → PDFProcessor, image → ImageProcessor, etc.).
3. Nested archives are extracted recursively up to `max_depth=3`.
4. Unknown file types are skipped; all `source` fields reference the original archive path.
5. Each file type follows its own processing pipeline below.
6. **URL archives**: when ingested via URL, archives are extracted from the temp download without copying to PVC; the original URL is the canonical source.

---

## PVC Preprocessing

Before format-specific processing, oversized media files are normalized **on disk** to bound storage and embedding costs:

| Type | Limit | Action |
|------|-------|--------|
| Image | `1920×1080` pixels | Downscaled (LANCZOS, aspect-preserving) via PIL; saved as `*_preprocessed` |
| Video | `1280×720` @ 24 fps | Transcoded via ffmpeg (libx264 CRF 28, AAC 128k, `+faststart`); saved as `*_preprocessed` |
| Audio | 5 MiB | Split into equal-duration segments via ffmpeg segment muxer (`-f segment -segment_time`); each segment becomes a separate document |

This is distinct from the per-doc `max_pixels=720×720` limit applied at embedding time (both apply — PVC stores at 1920×1080, embedder resizes to 720×720).

---

## Format-by-Format Processing

### PDF — Page-by-Page Text + Chart/Figure Extraction + Noise Filtering

**Library:** PyMuPDF (`fitz`)

**Three extraction levels:**

| Level | Output | Use |
|-------|--------|-----|
| `extract_pages()` | Per-page text + all embedded images as data URLs | Simple page-by-page |
| `extract_text_blocks()` | Block-level text with bounding boxes + nearby image detection | Spatial layout analysis |
| `extract_chunks()` | Chunked text with attached images (full list) | Backward-compatible ingestion |
| `extract_chunks_iter()` | **Generator** — yields chunks page-by-page | **Primary ingestion method** |

**Chunking flow:**
1. Each page is extracted via `page.get_text()` for text and `page.get_images()` with `doc.extract_image()` for embedded images.
2. Text blocks are grouped up to `chunk_size` tokens (default 2048) per chunk.
3. The last `chunk_overlap` tokens (default 256) of the previous chunk are carried forward into the next (across pages).
4. **Image proximity**: images within `_HORIZONTAL_PROXIMITY_PX=60` or `_VERTICAL_PROXIMITY_PX=120` of a text block are attached to that chunk as `image` entries (data URLs).
5. Standalone image blocks (no nearby text) are emitted as separate documents with `[Image on page N]` as text.
6. Images are deduplicated across chunks on the same page.
7. **Noise filtering**: reference lists, author lists, and tables of contents are auto-detected via heuristics and skipped.
8. Output: `{"text": "...", "image": ["data:image/...;base64,..."], "source": "...", "page": N}`

**Incremental extraction**: `extract_chunks_iter()` is a generator that opens the PDF once and yields chunks page-by-page. Overlap state carries between pages so cross-page chunking is identical to the list version. The batch ingestion pipeline uses this generator to hand off sub-batches to the embedding consumer **while later pages are still being extracted**, enabling concurrent extraction + embedding for large PDFs (e.g. 8k+ pages).

---

### Image — Resize → Data URL

**Library:** PIL (Pillow)

**Flow:**
1. Read local file bytes; guess MIME type via `mimetypes`.
2. Resize with PIL LANCZOS downscale so `width × height ≤ max_pixels` (default 720×720), preserving aspect ratio.
3. Encode as `data:{mime};base64,{b64}` data URL.
4. Text default: `[Image: {filename}]`.
5. HTTP(S) URLs are downloaded to temp files and processed through full resizing (remote URLs are not passed through untouched).
6. Output: `{"text": "[Image: photo.jpg]", "image": "data:image/jpeg;base64,...", "source": "/path/to/photo.jpg"}`
7. **Media persistence**: inline data URLs are saved to PVC files (`_save_doc_media()`) with a 1920×1080 pre-resize, then referenced via `file://` paths.

---

### Video — Overlapping Segments, Transcode

**Dependencies:** `ffmpeg` (system binary), `ffprobe`

**Parameters:** `fps=1.0, max_pixels=720×720, segment_seconds=32, overlap_seconds=4`

**Flow:**
1. Segment the video into overlapping windows: stride = `segment_seconds - overlap_seconds` = 28s (32s window, 4s overlap).
2. Segments shorter than `overlap_seconds` are skipped.
3. Each segment is transcoded **in memory** (no temp files) via ffmpeg pipe:
   ```
   ffmpeg -v error -ss {start} -t {duration} -i {input}
          -vf fps={fps},scale={new_w}:{new_h}
          -f mp4 -movflags frag_keyframe+empty_moov
          -vcodec libx264 -preset fast -crf 28 -
   ```
4. **Pixel budget**: If `total_pixels > 0`, the effective per-frame max is `min(max_pixels, total_pixels / num_frames)` to stay within the VLM's total pixel limit.
5. Scale is computed dynamically from the source aspect ratio.
6. Text format: `[Video: {name}] [{start}s – {end}s]`
7. Output: `{"text": "[Video: game.mkv] [0s – 32s]", "video": "data:video/mp4;base64,...", "source": "...", "timestamp_start": 0.0, "timestamp_end": 32.0}`

---

### Audio — Segmentation, Base64 Data URL

No dedicated processor file; handled inline in `dataset_manager.py`.

**Flow:**
1. Read raw file bytes.
2. Guess MIME type via `mimetypes`.
3. Base64 encode to `data:{mime};base64,{b64}`.
4. **Segmentation**: files larger than 5 MiB are split via ffmpeg's segment muxer into byte-budgeted segments; each segment becomes a separate document with `segment_index` and a label like `"foo.mp3 — segment 1/3"`.
5. Text: `[Audio: {filename}]`
6. Output: `{"text": "[Audio: recording.mp3]", "audio": "data:audio/mpeg;base64,...", "source": "/path/to/audio.mp3"}`

---

### Text / Markdown — Markdown-Aware Chunking

**Parameters:** `chunk_size` (dynamic, default 2048), `chunk_overlap` (default 256), `strip_markdown=False`

**Flow:**
1. **Heading detection**: splits on `^#{1,6}\s+(.+)$` to find section boundaries.
2. Small sections under the same parent heading are merged; oversized sections are split further.
3. **`_split_oversized()`**: tries paragraph boundaries first, falls back to character-level with word-boundary awareness.
4. **`_split_by_chars()`**: character-level split with overlap, breaking at word boundaries when possible.
5. **`_strip_md_formatting()`**: optionally removes `**bold**`, `__italic__`, `[links](url)`, `![images]`, `> blockquotes`, `- lists`, `| tables`, etc.
6. Output: `{"text": "...", "source": "...", "chunk_index": N}`

---

### JSON — Flatten → Key:Value Lines, Chunk by Top-Level Keys

**Parameters:** `code_chunk_size` (dynamic, default 8192), `code_chunk_overlap` (default 512), `flatten=True`, `separator="."`

**Flow:**
1. **Flatten**: `_flatten_json()` recursively converts nested JSON to `key.path: value` lines. Arrays are indexed numerically (`items.0.name`, `items.1.name`).
2. Top-level arrays: each array element becomes a separate document.
3. Large objects: chunked by top-level keys, grouping keys until `code_chunk_size` is reached.
4. Single oversized entries (e.g., a very long string value): split via `TextProcessor._split_text()`.
5. Metadata: `json_path` tracks the JSONPath expression (`$.records[0]`).
6. Output: `{"text": "name: Aurora\nlocation: Norway\n...", "source": "...", "json_path": "$.records[0]"}`

---

### Tables — Each Row → JSON String

| Format | Reader |
|--------|--------|
| `.csv`, `.tsv` | `csv.DictReader` (stdlib), UTF-8-BOM aware |
| `.xlsx`, `.xls` | `pandas.read_excel(engine='openpyxl')` |
| `.ods` | `pandas.read_excel(engine='odf')` |

**Parameters:** `chunk_size=8192, chunk_overlap=0, rows_per_doc=0`

**Flow:**
1. Each row is serialized as a compact JSON string: `json.dumps(row, ensure_ascii=False, default=str)`.
2. **Row grouping**:
   - `rows_per_doc > 0`: fixed-size row groups (e.g., 50 rows per document).
   - Default: character-budget grouping up to `chunk_size`, with optional `chunk_overlap` measured in rows.
3. Header row is repeated at the top of each group for context.
4. Output: `{"text": '{"Name": "Alice", "Age": "30"}\n\n{"Name": "Bob", "Age": "25"}', "source": "...", "row_index": N}`

---

### Code — Syntax-Aware Chunking, Function Boundaries

**Supported languages (16):** Python, JavaScript, TypeScript, Java, C++, C, C#, Go, Rust, Ruby, Swift, PHP, Kotlin, Scala, Shell, R (plus fallback patterns for unlisted extensions like `.pyw`, `.jsx`, `.mjs`, `.cjs`, `.tsx`, `.h`, `.hpp`).

**Parameters:** `code_chunk_size` (dynamic, default 8192), `code_chunk_overlap` (default 512), `add_language_annotation=True`

**Flow:**
1. **Language detection**: mapped from file extension; defines per-language regex patterns for top-level definitions:
   - **Python**: `def`, `class`, `@decorator`
   - **JS/TS**: `function`, `class`, `const/let/var = =>`, `export function`
   - **Java**: `class`, `interface`, `enum`, `record`, method signatures
   - etc.
2. **`_split_by_definitions()`**: splits code at definition boundaries. Each section is tagged with the definition name. False matches inside comments (`#`, `//`, `/*`) are skipped.
3. **`_build_chunks()`**: adjacent sections are merged up to `code_chunk_size`. Oversized single definitions (e.g., a 5000-line class) are split by line.
4. Optional `[Language: python]` annotation is prepended to each chunk.
5. Output: `{"text": "[Language: python]\ndef foo():\n    return 42\n", "source": "/path/to/file.py"}`

---

### Office Documents — Text + Inline Images

| Format | Library |
|--------|---------|
| `.docx` | `python-docx` |
| `.pptx` | `python-pptx` |
| `.odt`, `.odp` | `odfpy` |

**Parameters:** `chunk_size` (dynamic, default 2048), `chunk_overlap` (default 256)

**Flow:**

**DOCX:**
1. Extract inline images from relationships (`reltype == RT.IMAGE`).
2. Iterate paragraphs, traversing XML for `w:drawing` → `a:blip` to find inline images.
3. Tables are extracted as `cell1 | cell2 | ...` lines.
4. Paragraphs are grouped into chunks up to `chunk_size`; images are deduplicated.

**PPTX:**
1. Per-slide: collect all text frames and picture shapes (`MSO_SHAPE_TYPE.PICTURE`).
2. Each slide becomes one document with optional `image` list (data URLs).
3. Images are extracted via `shape.image.blob` → `data:{mime};base64,{b64}`.

**ODT/ODP:**
1. Text-only (no image extraction for ODF format).
2. Paragraphs and tables are extracted per-document / per-slide.

4. Output: `{"text": "...", "source": "...", "image": ["data:...", ...], "slide": N}`

---

### HTML — Tag Stripping, Heading-Aware Chunking

**Library:** BeautifulSoup (`html.parser`)

**Parameters:** `chunk_size` (dynamic, default 2048), `chunk_overlap` (default 256), `include_links=True`, `include_images=True`

**Flow:**
1. Parsed with BeautifulSoup; elements are converted to plain text:
   - `<p>` → plain text
   - `<ul>/<ol>` → `- item` list format
   - `<blockquote>` → `> text`
   - `<pre>` → verbatim
   - `<table>` → `| cell1 | cell2 |` format
   - `<img>` → `[Image: alt](src)` (when `include_images=True`)
   - `<a>` → `text (url)` (when `include_links=True`)
2. **Stripped**: `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<noscript>`.
3. **Heading-aware splitting**: splits at `<h1>`–`<h6>` boundaries; page title is prepended.
4. Chunking is delegated to `TextProcessor` (markdown-aware mode).
5. Output: `{"text": "Title: My Page\n\nHeading 1\n...", "source": "..."}`

---

### XML / YAML — Parse → Flatten → Key:Value Lines

**XMLProcessor:**
1. Parse via `xml.etree.ElementTree.fromstring()`.
2. Recursive `_xml_to_dict()` converts to nested dicts/lists:
   - Attributes prefixed with `@` (e.g., `@id="42"`).
   - Text content stored under `#text`.
   - Repeated tags become lists.
3. `_flatten_xml()` (reuses JSONProcessor's `_flatten_json()`) renders as `key: value` lines.
4. Large XMLs chunked by top-level children; oversized children split via `TextProcessor`.

**YAMLProcessor:**
1. `yaml.safe_load()` → parsed dict/list.
2. Delegates to `JSONProcessor.process_data()` with the same flatten/chunk logic.

3. Output: `{"text": "@id: 42\n#text: Hello\nchild.name: World\n", "source": "..."}`

---

### Jupyter Notebooks — Per-Cell Extraction

**Parameters:** `chunk_size` (dynamic, default 2048), `chunk_overlap` (default 256), `strip_markdown=False`, `include_code=True`, `include_outputs=True`

**Flow:**
1. **Markdown cells**: raw text (optionally strip markdown formatting).
2. **Code cells**:
   - Source wrapped in ` ```{lang}\n...\n``` `.
   - Outputs captured:
     - `stream` → stdout/stderr text
     - `execute_result` / `display_data` → text/plain/html/markdown + images (PNG/JPEG/GIF as data URLs)
     - `error` → `[Error: {ename}: {evalue}]`
3. **Raw cells**: source text as-is.
4. Consecutive cells of the same type are merged up to `chunk_size`; images accumulated.
5. Metadata: `cell_type` (`markdown`, `code`, `raw`).
6. Output: `{"text": "```python\nprint('hello')\n```\n\nhello", "source": "...", "image": ["data:image/png;base64,..."], "cell_type": "code", "chunk_index": N}`

---

### EPUB E-Books — Chapter Extraction, Inline Images

**Parameters:** `chunk_size` (dynamic, default 2048), `chunk_overlap` (default 256), `extract_images=True`

**Flow:**
1. Open EPUB as a zipfile; parse `META-INF/container.xml` to find the OPF manifest.
2. Read spine (reading order) from `<spine>` + `<itemref>`; map `id` → `href` from `<manifest>`.
3. For each spine item:
   - Read HTML content; strip HTML tags via regex (`<[^>]+>` → space).
   - Extract images: find `<img src="...">`, read from zip, convert to `data:{mime};base64,{b64}`.
   - Supported image MIME types: PNG, JPEG, GIF, WebP, SVG.
4. Chapters are merged up to `chunk_size`; images deduplicated.
5. Output: `{"text": "Chapter 1 text...", "source": "...", "image": ["data:image/png;base64,..."], "chunk_index": N}`

---

### Log Files — Timestamp/Severity Parsing, JSON-Lines Support

**Parameters:** `chunk_size=8192, chunk_overlap=0, max_entries_per_chunk=0`

**Flow:**
1. **Format detection**: auto-detects JSON-lines if >50% of the first 50 lines start with `{`.
2. **Structured log parsing** with regex patterns:
   - **Syslog**: `^(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)\s*(\S*)?\s*:?\s*(.*)`
   - **ISO-8601 timestamps**: `^\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}`
   - **Severity levels**: `TRACE`, `DEBUG`, `INFO`, `WARN(ING)`, `ERROR`, `FATAL`, `CRITICAL`
3. **JSON-lines**: each line parsed as JSON; extracts `message`/`msg`/`event` as primary text; `timestamp`/`time`/`ts`/`@timestamp` and `severity`/`level`/`loglevel` as metadata.
4. **Entry grouping**: new entry detection by timestamp/severity pattern matching; continuation lines merged.
5. **Chunking**: entries grouped by character budget or `max_entries_per_chunk`; overlapping supported.
6. Metadata: `timestamp_start`, `timestamp_end`, `severities` (set of observed severity levels).
7. Output: `{"text": "2024-01-01 12:00:00 ERROR Server crash\n...", "source": "...", "timestamp_start": "2024-01-01 12:00:00", "timestamp_end": "2024-01-01 12:00:05", "severities": ["ERROR"], "chunk_index": N}`

---

## Token-Based Chunking

When a HuggingFace tokenizer is bundled (`tokenizer_type="HuggingFace"`), all text/structured processors use **token counts** instead of character counts for chunk-size budgeting.

**`TokenTextSplitter`** (`utils/token_text_splitter.py`):
- Uses standalone HuggingFace `tokenizers` library (Rust, CPU-only — no PyTorch needed).
- `from_bundled()`: searches upward from `utils/` for `tokenizer.json`; returns `None` if missing (callers fall back to character-based chunking).
- `count_tokens()`: `len(tokenizer.encode(text).ids)`.
- `split_text()`: token-boundary splitting with 5% net-new tail avoidance (merges tiny final chunks into the previous).
- `merge_until_budget()`: merges fragments into groups fitting `chunk_size` tokens; carries overlap from the last fragment of the previous group.
- `overlap_text()`: returns last `chunk_overlap` tokens decoded back.

**Dual-budget system:**
- `EmbeddingModel.text_splitter` — uses `chunk_size`/`chunk_overlap` (defaults 2048/256).
- `EmbeddingModel.code_text_splitter` — uses `code_chunk_size`/`code_chunk_overlap` (defaults 8192/512).
- Structured types (code, json, xml, yaml) use the `code_*` splitter; everything else uses the general splitter.

**Bundled tokenizer**: The Dockerfile pre-downloads `tokenizer.json` from `huggingface.co/Qwen/Qwen3-VL-Embedding-8B` into `/app/tokenizer.json` at build time, enabling token-count chunking without runtime download.

---

## Embedding

**Model:** Qwen3-VL-Embedding-8B (4096-dimensional) — class `EmbeddingModel`

**Native modalities:** text, image, video

**Audio handling:** Audio is not natively supported by the embedder. The **Preprocessor** converts audio → text via ASR (Cohere Transcribe 03-2026) before embedding.

**Image/video handling:** Images and videos in retrieved documents can be passed through to the LLM natively (if the LLM supports them) or converted to text descriptions via a VLM (Gemma 4 31B) by the **Postprocessor**.

**Input conversion** (`InputConversion` in `langchain_overrides.py`):
- Accepts plain strings, bare media URLs (auto-detected), data URIs, local file paths, and dicts with `text`/`image`/`video`/`audio` keys.
- Bare-URL auto-detection supports `data:` URIs, local files (magic-byte sniffing via `_detect_media_type`), and remote URLs (via `mimetypes`).
- Media-only docs get an auto-inserted label like `[Image & Video media]` to avoid degenerate duplicate embeddings.
- Media fetched in parallel; video frames extracted client-side (PyAV-first, ffmpeg-subprocess fallback) when `max_video_frames > 0`.
- Images resized via PIL LANCZOS to `width × height ≤ max_pixels`.
- Conversational wrapper added (system/user/assistant) with instruction `"Represent the user's input."`.

**Text-only batch embedding**: Text-only documents (no media keys or unsupported modalities) are batched into a single `POST /v1/embeddings` request using the `input: [str1, str2, ...]` format. Each text is pre-formatted with the Qwen3-VL chat template on the client side (`_fmt_chat_template`), producing the same token sequence the server would generate from the `messages` format. This reduces N HTTP requests to 1 for text-only docs (typically ~60 of 64 docs per sub-batch). Multimodal docs still use individual `messages` requests, concurrent via `asyncio.gather`. Both paths run concurrently. Embeddings are >0.996 cosine similar to the per-doc `messages` format — within embedding noise.

**Embedding count guard**: After `aembed_documents`, the returned embedding count is checked against the document count. If they differ (e.g. API returned fewer vectors), a warning is logged and both lists are truncated to the shorter length to prevent silent data loss from `zip()`.

**Media processor kwargs** (shared with reranker): `fps=1.0, max_frames=64, min_pixels=4096, max_pixels=720×720, total_pixels=5×720×720`.

**Deduplication:**
- **File-level**: SHA-256 hash of input files; duplicates are skipped before copying to PVC. Tracked in `.hashes.json` per dataset.
- **Vector-level**: cosine similarity > 0.995 against existing vectors — skipped before insertion. Uses a **single batched `query_batch_points` call** per sub-batch (1 HTTP request instead of 64 individual queries). The `score_threshold` is enforced server-side by Qdrant, so only matches above the threshold are returned. The InMemoryVectorStore path uses vectorised numpy cosine similarity (N×M matrix in one shot). The threshold is tunable via the `RAG_DEDUP_THRESHOLD` env var.

**Media payload stripping:** After embedding and storage, base64 data URLs in Qdrant payloads are replaced with lightweight `file://` PVC paths to reduce storage size. Existing valid `file://` refs are left alone; remote URLs (`http://`/`https://`/`s3://`) are kept as-is.

**Batch ingestion**:
- Producer-consumer pipeline with background daemon thread.
- **Consumer crash resilience**: the consumer thread is wrapped in a top-level try/except. If it crashes unexpectedly, the error is propagated back to the caller as `{"status": "error", "error": ...}` instead of silently returning success with incomplete results.
- `batch_score=128.0` (2.56 MB ≈ 1.0 score) bounds embedding API payload size.
- **Generalized retry** (`retry_call` / `retry_async_call` in `general_tools.py`): 3 attempts with linear backoff; longer delays for connection errors. Used by S3 downloads, S3 prefix listing, and the embedding consumer.
- Progress callback events: `preprocessing`, `preprocessed` (includes `total` estimated chunk count), `embedding` (includes `chunks` sub-batch size), `complete`, `error` — streamed to clients via SSE.

**Sub-batched embed → dedup → upsert**:
- Documents are processed in sub-batches of ~64 (the embedder's `chunk_size`).
- Each sub-batch: embed → dedup → upsert → release. Embeddings and Document objects are freed after each upsert rather than accumulating for the full set.
- Media resize, audio splitting, and audio payload replacement happen **per sub-batch** (not upfront), bounding peak memory for large ingests.
- Data lands in Qdrant incrementally — queryable mid-ingest.

**Incremental PDF extraction**:
- `extract_chunks_iter()` generator yields chunks page-by-page.
- The batch producer hands sub-batches to the embedding consumer as soon as they're extracted, enabling concurrent extraction + embedding.
- For large PDFs (8k+ pages), embeddings start within seconds rather than waiting for full extraction.

---

## Retrieval

1. Query is embedded using the same Qwen3-VL-Embedding-8B model.
2. Cosine similarity search in Qdrant returns `top_k` (default 10) results.
3. Optional **Qwen3-VL-Reranker-8B** cross-encoder reranks the results; final count truncated to `reranker_top_k`.
4. **Score breakdown**: each result exposes `embedding_score` and `reranker_score` separately, in addition to the combined rounded `score`.

---

## Reranker (Separate Model Class)

The reranker is now a **first-class model role** (`RerankerModel`) cleanly separated from `EmbeddingModel`. It no longer carries embedding-specific fields (`embedding_dim`, `chunk_size`, `tokenizer_name`).

**Model:** Qwen3-VL-Reranker-8B — class `RerankerModel` → `MultiModalReranker`

**Two API endpoints** (under a base URL with `/v1` stripped, since `/score` and `/rerank` are non-standard OpenAI endpoints):

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/score` | POST `text_1` (query) + `text_2` (documents) | `list[list[float]]` (Q>1) or `list[float]` (Q=1) → raw relevance scores |
| `/rerank` | POST `query` + `documents` | `list[list[dict]]` → ranked dicts with `index` + `relevance_score` |

**RAG integration:** `_arerank_results` attaches both `_embedding_score` and `_reranker_score` to each doc dict, sorts by reranker score, truncates to `reranker_top_k`.

**Base URL handling:** `MultiModalReranker.__init__` strips `/v1` from the embedder's `base_url` to avoid OpenAI client warnings on the non-standard endpoints.

---

## LLM Generation

Two modes:

- **Text-only LLM (DeepSeek-V4-Flash):** All retrieved modalities are converted to text by the Postprocessor (image/video → VLM description, audio → ASR transcription) before being passed to the model.
- **Multimodal LLM (Gemma 4 31B):** Image and video are passed through natively. Only audio is converted to text.

A routing step can optionally skip RAG entirely if the LLM determines it can answer from its training data.

**RAG generation flow** (`MultiModalRAGSystem.agenerate`):
1. **Routing** (optional): asks the LLM "YES or NO" — if NO, answers directly without retrieval.
2. **Retrieval**: embeds query, similarity search, optional rerank.
3. **Postprocessing decision**: if `use_vlm=True` and retrieved docs contain media unsupported by the LLM → run `Postprocessor`.
4. **Multimodal content building**: assembles OpenAI-compatible content parts (text/image_url/video_url/audio_url); collapses to plain string when all text-only. `file://` paths are converted to inline `data:` URLs for LLM consumption.
5. **Generation**: builds messages (system prompt + user content), calls the LLM.

**Postprocessor fallback links**: When the LLM doesn't support a modality and the conversion model (ASR/VLM) is unavailable, the Postprocessor includes clickable HTTP links in the text so the LLM can share references with the user:
- `[Audio file]: {url}` when ASR is None
- `[Image file]: {url}` when VLM is None
- `[Video file]: {url}` when VLM is None

**Clickable source references**: All source references in context text and multimodal content use `_pvc_to_http_url()` to convert `file://` PVC paths to HTTP URLs via `MEDIA_BASE_URL`, so the LLM always sees clickable links rather than internal storage paths.

**Local vs remote models:** `model_usage` flag (`.remote()` / `.local()`) switches between in-cluster service DNS (`.svc.cluster.local`) and external URLs. Remote mode disables SSL verification and uses connection-pooled HTTP clients.

**MCP agent support:** Any `ChatModel` can build a LangChain agent with MCP tools via `agent()` / `aagent()`. Tools are loaded via `MultiServerMCPClient` from `langchain_mcp_adapters`.

---

## API Server

**17 REST endpoints** (`api_server.py`):

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/healthz` | Liveness/readiness |
| POST | `/api/datasets` | Create (name, description, caption_video, password) |
| POST | `/api/datasets/{name}/verify-password` | Verify password → 200/401/403 |
| GET | `/api/datasets` | List all |
| GET | `/api/datasets/{name}` | Get one (uses `X-Dataset-Password` header) |
| PATCH | `/api/datasets/{name}` | Update metadata (description, caption_video) |
| DELETE | `/api/datasets/{name}` | Delete dataset + Qdrant collection |
| POST | `/api/datasets/{name}/documents` | Add raw text/dict docs |
| POST | `/api/datasets/{name}/files` | Single file upload (multipart) |
| POST | `/api/datasets/{name}/batch-files` | Multi-file upload with **SSE progress streaming** |
| POST | `/api/datasets/{name}/batch-urls` | S3/HTTP URL ingestion with SSE streaming |
| GET | `/api/datasets/{name}/search` | Text search (`q`, `top_k`, `use_reranker`, `reranker_top_k`) |
| POST | `/api/datasets/{name}/search` | Multimodal search (body: text + image/video/audio) |
| GET | `/api/datasets/{name}/documents` | List stored docs (`limit`, default 50, max 1000) |
| DELETE | `/api/datasets/{name}/documents/{doc_id}` | Delete single doc |
| GET | `/api/datasets/{name}/files/{filepath:path}` | Serve stored file (password via header or query) |
| GET | `/api/admin/storage` | PVC disk usage + per-dataset stats |
| GET | `/` | HTML frontend |
| GET | `/manage` | HTML storage management view |

**Search parameters:** `q` (required), `top_k` (1–100, default 10), `use_reranker` (bool, default false), `reranker_top_k` (1–50, default 3).

---

## MCP Server

Exposes **9 MCP tools** (`mcp_server.py`):

| Tool | Purpose |
|------|---------|
| `list_datasets()` | Returns formatted text of all datasets (with `[caption_video]` / `[password]` markers) |
| `unlock_dataset()` | Verify a dataset password and cache the unlock for the session (default TTL 30 min) |
| `search_dataset()` | Multimodal search (`dataset_name`, `query`, `image`/`video`/`audio`, `top_k`, `use_reranker`, `reranker_top_k`, `password`, `media_base_url`). Instantiates a `Postprocessor` for modality conversion based on `base_llm_modalities`. |
| `get_dataset_files()` | List or retrieve files from a dataset (text inline; binary returns metadata + `download_url`) |
| `get_dataset_info()` | Returns dataset metadata |
| `describe_media()` | Standalone VLM description of an image/video (no dataset needed) |
| `transcribe_audio()` | Standalone ASR transcription of an audio file (no dataset needed) |
| `add_memory()` | Store an LLM-curated memory into a personal memory dataset. `dataset_name`/`password` optional — resolved from the `X-Memory-Dataset` / `X-Dataset-Password` request headers (or `MEMORY_DATASET` env) so the model does not pass them. Merges provenance metadata (`source`, `memory_kind`, `memory_ts`, `memory_tags`, `session_id`) into the Qdrant payload. |
| `search_memory()` | Recall from the personal memory dataset; same resolution + retrieval/postproc as `search_dataset` (via the shared `_run_retrieval` helper). |

**Long-term memory (opencode):** the `add_memory` / `search_memory` tools back a per-user long-term memory store. An MCP client (e.g. opencode) connects **twice** to the same URL — once as `rag-memory` (sending `X-Memory-Dataset`/`X-Dataset-Password` headers, exposing only `add_memory`/`search_memory`) and once as `rag-knowledge` (exposing the general dataset tools). Per-user isolation is the dataset **password**; the memory headers are read ONLY inside `add_memory`/`search_memory`, so a memory password can never silently unlock another dataset. See `MCP.md`, `MEMORY.md`, `opencode.jsonc`, and `AGENTS.md` for the full pattern.

**Transport:** Default `streamable-http` (port 9090 in helm). Also supports `stdio` and `sse`. A `_MemoryHeaderMiddleware` (wired in `main()`) captures the memory-identity headers into `contextvars.ContextVar`s for the memory tools.

**Query-vector caching:** `search_dataset`/`search_memory` reuse a stored Qdrant vector when the query media is already in the dataset, else a hash-keyed in-process LRU cache, avoiding re-embedding the same media twice.

**PVC→HTTP URL conversion:** When `media_base_url` is set (via `MEDIA_BASE_URL` env), `file://` PVC paths in results are rewritten to `{media_base_url}/api/datasets/{name}/files/{path}` HTTP URLs.

**DNS rebinding protection disabled** (`TransportSecuritySettings(enable_dns_rebinding_protection=False)`) for cluster networking.

---

## Security

**Dataset password protection:**
- PBKDF2-SHA256, 100,000 iterations, random 16-byte hex salt.
- Stored in `meta.json` as `password_hash`.
- Verified via `X-Dataset-Password` header or `?password=` query param (for media tags).
- API responses strip the hash and add a `has_password` boolean.
- Methods: `create_dataset(password=)`, `has_password()`, `verify_password()`, `set_password()`.

---

## Ingestion Sources

Beyond local file uploads, the system ingests from:

- **S3 URLs** (`s3://bucket/key`): uses boto3 for download via `_get_s3_client()`.
- **S3 directory prefixes** (`s3://bucket/prefix/`): lists all supported-type objects under the prefix and ingests them as a batch.
- **HTTP(S) URLs**: downloaded to temp files and processed through the full pipeline (including resizing).

All remote sources are handled via `add_urls_batch()` with SSE progress streaming and the same retry/batch-score logic as file uploads.

**S3/MinIO configuration**: The S3 client reads connection details from environment variables, supporting both AWS S3 and on-cluster MinIO:
- `S3_ENDPOINT_URL` — custom endpoint (e.g. `http://minio.minio.svc.cluster.local:9000`), stored in ConfigMap
- `S3_ACCESS_KEY_ID` — access key, stored in Kubernetes Secret
- `S3_SECRET_ACCESS_KEY` — secret key, stored in Kubernetes Secret

When `S3_ENDPOINT_URL` is unset, the default boto3 credential chain (IAM roles, `~/.aws/credentials`) is used. The Helm `s3` section is optional — excluding it leaves the env vars empty and falls back to default credentials.

**Retry logic**: S3 downloads (`_download_s3`) and prefix listings (`_list_s3_prefix`) use `retry_call` with 3 attempts and linear backoff (longer for connection errors), matching the embedding consumer's retry behavior.

---

## Storage & Metadata

**Per-dataset metadata** (`meta.json`): `name`, `description`, `caption_video`, `created`, `document_count`, `password_hash`, `file_type_counts` (dict of per-type doc counts).

**Dataset management:**
- Each dataset = one Qdrant collection named after the dataset.
- Document counts are synced from Qdrant's actual point count on read.
- `file_type_counts` tracks per-type document counts (e.g., `{"pdf": 42, "image": 13}`).
- `original_source` field preserves the original basename alongside the stored `source` PVC path.
- Document deletion by Qdrant point ID (`PointIdsList` selector with `wait=True`).

**Endpoint verification** (at startup): all 4 model endpoints (embedder, reranker, vlm, asr) are pinged via OpenAI-compatible `GET /v1/models`. Raises `RuntimeError` if unreachable.

**Admin storage stats:** `/api/admin/storage` returns PVC disk usage (total/used/free/utilization) and per-dataset breakdowns with file-type sub-items (docs + bytes per type). Runs in a thread pool (via `run_in_executor`) so health/readiness probes stay responsive during large collection scans. File-type backfill uses paginated Qdrant scroll (256 points at a time, capped at 50k) with `PayloadSelectorInclude` to fetch only `metadata.source` instead of full payloads — preventing memory spikes and event-loop blocking that previously caused pod crashes on the management page.

---

## Batched HTTP Calls

The ingestion pipeline minimises HTTP round-trips to external services by batching at every layer:

| Call | Batch method | HTTP requests per sub-batch (~64 docs) |
|------|-------------|---------------------------------------|
| **Embedding API (text-only)** | `input: [chat_template_text1, ...]` — 1 POST with all text-only docs | 1 (was 64) |
| **Embedding API (multimodal)** | Individual `messages` POSTs, concurrent via `asyncio.gather` | N (N = # of multimodal docs, typically ~4) |
| **Qdrant dedup** | `query_batch_points` with server-side `score_threshold` — 1 POST for all embeddings | 1 (was 64) |
| **Qdrant upsert** | `client.upsert(points=[...])` — all points in one call | 1 |
| **Reranker** | `/rerank` POST with all documents per query; multiple queries concurrent via `asyncio.gather` | 1 per query |
| **Qdrant payload stripping** | `client.retrieve(ids=[...])` + grouped `client.set_payload` by unique payload | 1 retrieve + N set_payload (N = unique payload groups) |

Text-only and multimodal embedding paths run **concurrently** via `asyncio.gather(text_task, mm_task)` — while the text batch is processed by the GPU, the multimodal path fetches media and fires off individual requests in parallel.

For a typical PDF sub-batch (~60 text + ~4 image docs): **~5 HTTP requests** instead of 64. For the Stacks project (~8k chunks, mostly text): ~625 requests instead of ~8,000.

---

## Edge Case Fixes (v0.4.0)

### PDF Processor (`pdf_processor.py`)

| Fix | Impact |
|-----|--------|
| **All-image page flush** | Pages with only images (no text blocks) now flush `current_text`/`current_images` before `continue` — cross-page overlap was silently dropped |
| **`_MAX_IMAGES_PER_CHUNK` cap fix** | Images beyond the 20-per-chunk cap are now re-emitted as standalone chunks instead of silently lost |
| **CMYK → RGB conversion** | `n==4` (CMYK) images now converted to RGB via `fitz.Pixmap(fitz.csRGB, pix)` — was producing unreadable CMYK PNGs |
| **`_extract_page_blocks` exception logging** | Image extraction failures now log a warning with image index and page number instead of silent `pass` |
| **Bbox float comparison** | Image block matching uses `_bbox_close()` with 1.0pt tolerance instead of exact float equality |
| **Degenerate image filtering** | Images <10×10px or <1000 total pixels filtered out; max 20 images per chunk — fixes 400 errors from hundreds of 541×1 horizontal-rule images |

### General Processors

| Fix | File | Impact |
|-----|------|--------|
| **`<hr>` element** | `html_processor.py` | `text == "\n---\n"` (comparison) → `text = "\n---\n"` (assignment) — horizontal rules were silently dropped |
| **JSON arrays in logs** | `log_processor.py` | JSON-lines parser checks `isinstance(obj, dict)` before `.items()` — arrays/scalars crash fixed |

### API / Pipeline

| Fix | File | Impact |
|-----|------|--------|
| **Temp file leak** | `api_server.py` | Batch-files temp files cleaned up in `finally` block on success path (was only cleaned on error) |
| **Consumer thread crash** | `dataset_manager.py` | Consumer thread wrapped in top-level try/except; errors propagated as `{"status": "error"}` instead of silent data loss |
| **Embedding count mismatch** | `rag_system.py` | After `aembed_documents`, checks `len(embs) != len(docs)` — logs warning and truncates to prevent `zip()` silently dropping docs |
| **Qdrant outage** | `rag_system.py` | `get_collections()` wrapped with `retry_call(max_attempts=3)` — retries instead of crashing on transient Qdrant unavailability |
| **Single-file upload crash** | `api_server.py` | Catches `FileNotFoundError` (404) and broad `Exception` (500) in addition to `ValueError` (400) |

---

## Upload UX

**XHR upload with progress**: Uses `XMLHttpRequest` instead of `fetch` for real upload progress bars (`📤` → `📦` → `⚙` → `◐` → spinner → `✓`).

**Chunk count accumulation**: Per-file chunk counts are accumulated across sub-batches via `_fileChunkTotal` map. A file split across multiple sub-batches shows its total chunk count (e.g., `✓ 2,048 chunk(s)`) instead of just the last sub-batch's count.

**Estimated chunk totals**: The "preprocessed" progress event includes `total` (estimated total chunks for the file). During embedding, the line shows `128 / (est.) 1,024 chunks embedded…` once the total is known. The estimate arrives after the generator finishes extracting the file — the frontend updates from `128 chunks embedded…` to `128 / (est.) 1,024 chunks embedded…` when it arrives.

**Connection drop resilience**: `xhr.onerror` and `xhr.onload` poll `GET /api/datasets/{name}` every 10s for `document_count`. Shows "Processing in background" instead of "Failed". Marks complete when count stabilizes (3 unchanged polls).

**Timer fix**: The upload timer `setInterval` is cleared in `xhr.upload.onload` so the SSE-driven chunk count summary isn't overwritten with `"Uploading 14 file(s)... 219s"` after upload completes.

---

**Environment variables** (`model_config.py`):
```
MODEL_<ROLE>_NAME       — model name
MODEL_<ROLE>_URL        — full remote URL
MODEL_<ROLE>_API_KEY    — API key / service-account token
MODEL_<ROLE>_CLASS      — "MultiModalEmbeddings" | "MultiModalReranker" | "ChatOpenAI"
MODEL_<ROLE>_EXTRA      — JSON of additional constructor kwargs
```

**Four roles:** `EMBEDDER`, `RERANKER`, `VLM`, `ASR`.

**S3 / MinIO** (optional):
```
S3_ENDPOINT_URL       — custom endpoint (e.g. http://minio.minio.svc.cluster.local:9000)
S3_ACCESS_KEY_ID      — access key
S3_SECRET_ACCESS_KEY  — secret key
```

Factory functions `build_embedder()`, `build_reranker()`, `build_vlm()`, `build_asr()`, `build_all()` construct model instances from env vars, falling back to `None` for any missing role.

**Available models** (`pcai_models.py`):

| Model | Class | Purpose |
|-------|-------|---------|
| `deepseek_v4_flash_280B` | ChatModel | Default text LLM |
| `qwen35_397B` | ChatModel | Large MoE LLM |
| `gemma4_31B` | ChatModel | Multimodal VLM (default) |
| `nemotron_3_super_120B` | ChatModel | LLM (not deployed) |
| `minimax_2_7_240B` | ChatModel | LLM (not deployed) |
| `qwen3_vl_8B` | EmbeddingModel | Multimodal embeddings (default) |
| `qwen3_vl_reranker_8B` | RerankerModel | Cross-encoder reranker (default) |
| `whisper_large_v3_turbo` | VoiceModel | ASR (not deployed) |
| `cohere_transcribe_3_2b` | VoiceModel | ASR (default) |
| `qwen3_tts_1_7B` | VoiceModel | TTS with 9 supported voices |

**Deployment** (Helm chart): 2-container pod (API server port 8000 + MCP sidecar port 9090), Qdrant StatefulSet with dedicated PVC, PCAI VirtualService on `istio-system/ezaf-gateway` (timeout 660s), AuthorizationPolicy via `oauth2-proxy`, Kyverno pod security policy.
