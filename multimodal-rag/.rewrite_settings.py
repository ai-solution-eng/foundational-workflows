import re
PATH = "src/multimodal_rag/templates/index.html"
html = open(PATH).read()
start = html.index("D23 (UX): dataset settings")
start = html.rindex("<!--", start - 120, start + 40)
doc_list = html.index('<div id="doc-list"', start)
seg = html[start:doc_list]
end = start + seg.rindex("</div>") + len("</div>\n")
block = """<!-- D23 (UX, revised 2026-10): dataset settings as a nested list with ONE
     Save button.  saveDatasetAll() applies the three PATCHes in order;
     per-section failures are named in one combined result line. -->
    <div style="margin-top:12px;border:1px solid var(--border);border-radius:6px;padding:10px 12px;background:var(--surface-2)">
      <div style="font-size:.78rem;font-weight:600;color:var(--text-muted);margin-bottom:4px;letter-spacing:.03em">DATASET SETTINGS</div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-bottom:8px">Ingest rows affect <b>future uploads</b> only; RRF and Public apply immediately.</div>
      <div style="font-size:.85rem">
        <div style="font-weight:600;margin-bottom:4px">Ingest behavior</div>
        <div style="margin-left:16px;display:grid;grid-template-columns:auto 1fr;gap:4px 12px;align-items:center">
          <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-weight:400;white-space:nowrap" title="Transcribe video audio tracks via ASR for NEW uploads to this dataset.">Caption with ASR <input type="checkbox" id="edit-caption-asr"></label>
          <span style="color:var(--text-muted);font-size:.72rem">ASR transcripts for video audio</span>
          <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-weight:400;white-space:nowrap" title="VLM-describe images/videos at ingest for NEW uploads. Enables VLM-skip at retrieval.">Caption with VLM <input type="checkbox" id="edit-caption-vlm"></label>
          <span style="color:var(--text-muted);font-size:.72rem">VLM descriptions for images/videos</span>
          <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-weight:400;white-space:nowrap" title="OCR fallback via tesseract for scanned pages (images, no text layer) for NEW uploads to this dataset.">OCR fallback <input type="checkbox" id="edit-ocr"></label>
          <span style="color:var(--text-muted);font-size:.72rem">scanned pages (no text layer) only</span>
          <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-weight:400;white-space:nowrap" title="Contextual retrieval (ingest-time document context per chunk) for NEW uploads to this dataset. Affects new ingests only - Recreate to re-contextualize existing files.">Contextual retrieval <input type="checkbox" id="edit-contextual"></label>
          <span style="color:var(--text-muted);font-size:.72rem">document context per chunk — Recreate applies it retroactively</span>
        </div>
        <div style="font-weight:600;margin:10px 0 4px">Reciprocal Rank Fusion (RRF) — hybrid search defaults <span style="font-weight:400;color:var(--text-muted);font-size:.72rem">(query-time, immediate)</span></div>
        <div style="margin-left:16px;display:grid;grid-template-columns:auto 1fr;gap:4px 12px;align-items:center">
          <label style="display:inline-flex;align-items:center;gap:6px;font-weight:400;white-space:nowrap" title="Rank-space tilt for the dense (vector) lane, NOT a score multiplier: a higher weight makes that lane's rank positions count more. Blank = 1.0.">Dense weight <input type="number" id="edit-rrf-dense" step="0.001" min="0" max="10" placeholder="1.0" style="width:80px;padding:4px 8px;border:1px solid var(--border);border-radius:4px;background:var(--surface-2);color:var(--text);font-size:.85rem"></label>
          <span style="color:var(--text-muted);font-size:.72rem">vector lane — how much its rank order counts (0 = lean keyword-only)</span>
          <label style="display:inline-flex;align-items:center;gap:6px;font-weight:400;white-space:nowrap" title="Rank-space tilt for the sparse (BM25 keyword) lane. Blank = 1.0.">Sparse weight <input type="number" id="edit-rrf-sparse" step="0.001" min="0" max="10" placeholder="1.0" style="width:80px;padding:4px 8px;border:1px solid var(--border);border-radius:4px;background:var(--surface-2);color:var(--text);font-size:.85rem"></label>
          <span style="color:var(--text-muted);font-size:.72rem">BM25 keyword lane — 0 = lean vector-only, 2+ = keyword-heavy</span>
          <label style="display:inline-flex;align-items:center;gap:6px;font-weight:400;white-space:nowrap" title="RRF ranking constant k (Qdrant default 2): how sharply the top ranks dominate the fusion. Lower k (1) flattens the curve — deeper ranks count more; higher k (5-10) concentrates weight on the very top results. Leave blank for the default.">Ranking constant k <input type="number" id="edit-rrf-k" step="1" min="1" max="1000" placeholder="2 (default)" style="width:85px;padding:4px 8px;border:1px solid var(--border);border-radius:4px;background:var(--surface-2);color:var(--text);font-size:.85rem"></label>
          <span style="color:var(--text-muted);font-size:.72rem">top-rank sharpness — lower flattens, higher concentrates</span>
        </div>
        <div style="font-weight:600;margin:10px 0 4px">Visibility</div>
        <div style="margin-left:16px;display:grid;grid-template-columns:auto 1fr;gap:4px 12px;align-items:center">
          <label id="edit-public-row" style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-weight:400;white-space:nowrap" title="Public (GLOBAL): every minted API key / SSO user can list, read and search this dataset without an explicit grant. Creator-or-admin only (D23). Password-protected datasets cannot be made public.">Public (all users) <input type="checkbox" id="edit-public-flag"></label>
          <span style="color:var(--text-muted);font-size:.72rem" id="public-note">listed for every user; self-selectable without a password</span>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;gap:8px;align-items:center">
        <button class="btn btn-primary btn-sm" onclick="saveDatasetAll()" title="Apply ALL settings in one pass">Save settings</button>
        <span id="settings-note" style="font-size:.8rem;color:var(--text-muted)"></span>
      </div>
    </div>
"""
html2 = html[:start] + block + html[end:]
open(PATH, "w").write(html2)
ids = ["edit-caption-asr","edit-caption-vlm","edit-ocr","edit-contextual","edit-rrf-dense","edit-rrf-sparse","edit-rrf-k","edit-public-flag","public-note","settings-note","doc-list"]
for i in ids:
    assert len(re.findall('id="%s"' % i, html2)) == 1, i
print("OK: block replaced, 11 IDs intact")
