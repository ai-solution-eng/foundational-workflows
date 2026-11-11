"""Folder uploads in the upload tab: template controls and the directory walker.

``index.html`` is a single-file frontend, so the folder intake lives in the
template. Two things are covered here, mirroring the existing template gates
(RRF controls, contextual controls):

1. the template ships the folder controls — a ``webkitdirectory`` picker, the
   Upload Folder button, and the drop-path expansion; and
2. the directory walker actually walks — nested folders, ``readEntries``
   batching (the API answers at most one bounded batch per call), skipped dot
   entries and unreadable members — exercised in Node against fake entries.

Run::

    pytest tests/full_pipeline/test_folder_upload_ui.py
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[2] / "src" / "multimodal_rag" / "templates" / "index.html"

# The extracted section is delimited by two banners the template already
# carries: the folder-intake banner and the retry-tracking declaration that
# follows it. A moved layout fails the marker assertions instead of silently
# testing an empty string.
SECTION_START = "// Folder intake"
SECTION_END = "// Track files for retry"


def _template_text() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def test_template_ships_the_folder_upload_controls() -> None:
    """UI smoke: the folder picker, its button and the drop-path expansion ship."""
    html = _template_text()
    for required in (
        'id="folder-input"',
        "webkitdirectory",
        'id="pick-folder-btn"',
        "pickFolder()",
        "onFolderPicked(this)",
        "_droppedItems(e.dataTransfer)",
        "handleDroppedEntries(entries, loose)",
        "async function _walkDirectory",
        "Nothing to upload — no readable files were found",
        "async function _queueUploads",
    ):
        assert required in html, f"template must contain {required!r}"

    # The picker sits OUTSIDE the drop zone: the zone's click handler opens the
    # file dialog, and a programmatic click on a child input would bubble back
    # into that handler.
    assert html.index('id="folder-input"') > html.index('id="file-input"')

    # A folder is expanded into files and then enters the ordinary pipeline —
    # the intake must not gain a second upload path of its own.
    assert "await handleFiles(files)" in html


def test_directory_walker_expands_folders() -> None:
    """Behaviour: the walker collects nested files, drains batches, skips junk."""
    node = shutil.which("node")
    if node is None:
        print("  (node not available — skipping the walker behaviour check)")
        return

    html = _template_text()
    start = html.index(SECTION_START)
    end = html.index(SECTION_END)
    assert start < end, "folder-intake section markers are out of order"
    section = html[start:end]
    assert "async function _walkDirectory" in section, "extracted section lost the walker"

    harness = """
'use strict';
// ---- stubs the extracted section closes over -----------------------------
const selectedDataset = 'ds';
const toasts = [];
function toast(message) { toasts.push(message); }
const uploaded = [];
async function handleFiles(files) { for (const f of files) uploaded.push(f.name); }
let confirmCalls = 0;
function confirm() { confirmCalls += 1; return true; }

__SECTION__

// ---- fake entry API ------------------------------------------------------
function fileEntry(name, unreadable = false) {
  return {
    name, isFile: true, isDirectory: false,
    file(ok, fail) { if (unreadable) fail(new Error('gone')); else ok({ name, size: 1 }); },
  };
}
function dirEntry(name, children, batchSize = 100) {
  return {
    name, isFile: false, isDirectory: true,
    createReader() {
      let index = 0;
      return {
        readEntries(ok) {
          const batch = children.slice(index, index + batchSize);
          index += batch.length;
          ok(batch);
        },
      };
    },
  };
}

function assertEqual(actual, expected, what) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) throw new Error(what + ': expected ' + e + ', got ' + a);
}

async function main() {
  // 1. Nested walk: skips dot entries, Office lock files and OS metadata, and
  //    drops a member the browser refuses to read.
  const project = dirEntry('project', [
    fileEntry('readme.md'),
    fileEntry('~$draft.docx'),
    fileEntry('Thumbs.db'),
    fileEntry('broken.pdf', true),
    dirEntry('.git', [fileEntry('config')]),
    dirEntry('docs', [
      fileEntry('guide.pdf'),
      dirEntry('nested', [
        dirEntry('2024', [
          dirEntry('q1', [fileEntry('deep.txt')]),
        ]),
      ]),
      dirEntry('empty', []),
    ]),
  ]);
  const walked = [];
  await _walkDirectory(project, walked);
  //     (docs/nested/2024/q1 is four directories below the dropped root)
  assertEqual(walked.map(f => f.name), ['readme.md', 'guide.pdf', 'deep.txt'], 'nested walk');

  // 2. readEntries answers one bounded batch per call: 250 members must all be
  //    collected, which only happens if the reader is drained to EOF.
  const many = [];
  for (let i = 0; i < 250; i++) many.push(fileEntry('f' + i + '.txt'));
  const bulk = [];
  await _walkDirectory(dirEntry('bulk', many, 100), bulk);
  assertEqual(bulk.length, 250, 'batch drain');

  // 3. A drop mixes loose files with folders; both arrive at the pipeline.
  uploaded.length = 0;
  await handleDroppedEntries([project], [{ name: 'loose.png' }]);
  assertEqual(uploaded, ['loose.png', 'readme.md', 'guide.pdf', 'deep.txt'], 'mixed drop');

  // 3b. The root the user dropped is taken as-is: an explicitly dropped .env
  //     is uploaded, not treated as a hidden entry to skip.
  uploaded.length = 0;
  await handleDroppedEntries([fileEntry('.env')], []);
  assertEqual(uploaded, ['.env'], 'explicit top-level file');

  // 4. An empty folder reports instead of uploading nothing silently.
  toasts.length = 0;
  await _queueUploads([]);
  assertEqual(toasts, ['Nothing to upload — no readable files were found'], 'empty pick notice');

  // 5. A picker path skips when any segment BELOW the picked root is junk.
  assertEqual(_skipRelativePath('project/.git/HEAD'), true, 'dot directory path');
  assertEqual(_skipRelativePath('project/docs/guide.pdf'), false, 'ordinary path');
  assertEqual(_skipRelativePath('.config/settings.json'), false, 'picked root is taken as-is');
  assertEqual(_skipRelativePath('project/docs/nested/2024/q1/deep.txt'), false, 'deep nested path');

  // 6. A large pick is confirmed before it is queued.
  confirmCalls = 0;
  uploaded.length = 0;
  const big = [];
  for (let i = 0; i < 1001; i++) big.push({ name: 'big' + i + '.txt' });
  await _queueUploads(big);
  assertEqual(confirmCalls, 1, 'large pick confirmed');
  assertEqual(uploaded.length, 1001, 'large pick queued');

  console.log('walker ok');
}

main().then(
  () => process.exit(0),
  (err) => { console.error(err && err.stack ? err.stack : String(err)); process.exit(1); },
);
""".replace("__SECTION__", section)

    workdir = Path(tempfile.mkdtemp(prefix="folder-upload-"))
    script = workdir / "walker-check.js"
    script.write_text(harness, encoding="utf-8")
    try:
        proc = subprocess.run(
            [node, str(script)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    assert proc.returncode == 0, f"walker check failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "walker ok" in proc.stdout
