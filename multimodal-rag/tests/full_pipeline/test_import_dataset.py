"""Offline tests for backup restore / import (roadmap feature 4).

Covers :meth:`DatasetManager.prepare_import` / ``replay_imported_documents``
/ ``import_dataset`` against hand-built export archives (the same member
layout ``GET /api/datasets/{name}/export`` writes):

  * files mode — ``files/`` extracted under the target dataset, meta.json
    restored (password hash / embedder fingerprint / counters stripped),
    re-embed delegated to ``recreate_dataset``.
  * text-replay mode — raw-documents backups (no ``files/``): rows'
    ``page_content`` re-embedded, bare media placeholders skipped, media
    payload keys stripped, lightweight metadata preserved.
  * collisions (``FileExistsError`` unless ``overwrite=True``), password
    restore, and traversal-member rejection.

No Qdrant server, no models — the slow phases are stubbed.

Run::

    python tests/full_pipeline/test_import_dataset.py    # standalone
    pytest tests/full_pipeline/test_import_dataset.py    # under pytest
"""

import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.dataset_manager import DatasetManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dm(tmpdir: Path) -> tuple[DatasetManager, dict[str, Any]]:
    """A bare DatasetManager with stubbed Qdrant/slow phases.

    ``replay_imported_documents`` is deliberately NOT stubbed — the replay
    tests exercise the real method end-to-end (it only needs the
    ``add_documents`` stub).
    """
    dm = DatasetManager.__new__(DatasetManager)
    dm.base_path = tmpdir
    dm.datasets_path = tmpdir / "datasets"
    dm.datasets_path.mkdir(parents=True, exist_ok=True)
    recorded: dict[str, Any] = {"deleted": [], "recreated": [], "added": []}

    dm._invalidate_has_password = lambda name: None  # type: ignore[method-assign]
    dm.delete_dataset = lambda name: recorded["deleted"].append(name)  # type: ignore[method-assign]
    dm.recreate_dataset = (  # type: ignore[method-assign]
        lambda name, file_entries=None, progress_callback=None: (
            recorded["recreated"].append(name),
            {"status": "ok", "file_count": len(file_entries or []) or 2},
        )[1]
    )

    def _add_documents(ds_name: str, docs: list[dict]) -> list[str]:
        recorded["added"].append(list(docs))
        return [f"id{i}" for i in range(len(docs))]

    dm.add_documents = _add_documents  # type: ignore[method-assign]
    return dm, recorded


def _build_export(
    tar_path: Path,
    *,
    name: str = "ds1",
    files: dict[str, bytes] | None = None,
    rows: list[dict] | None = None,
    protected: bool = False,
) -> None:
    meta = {"name": name, "description": "test export", "document_count": 42, "created": "2026-08-30T00:00:00"}
    if protected:
        meta["password_hash"] = "pbkdf2$sha256$..."
    if rows is None:
        rows = []
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("meta.json")
        payload = json.dumps(meta).encode()
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        info = tarfile.TarInfo("documents.jsonl")
        payload = "\n".join(json.dumps(r) for r in rows).encode()
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        for fname, data in (files or {}).items():
            info = tarfile.TarInfo(f"files/{fname}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tar_path.write_bytes(buf.getvalue())


# ---------------------------------------------------------------------------
# prepare_import — files mode
# ---------------------------------------------------------------------------


def test_prepare_import_files_mode():
    with tempfile.TemporaryDirectory() as td:
        dm, _recorded = _make_dm(Path(td))
        tar = Path(td) / "backup.tar.gz"
        _build_export(tar, name="ds1", files={"a.pdf": b"%PDF-fake", "sub/b.txt": b"hello"})

        plan = dm.prepare_import(tar)
        assert plan["dataset"] == "ds1"
        assert plan["mode"] == "re-embed-files"
        assert plan["file_count"] == 2
        assert plan["had_password"] is False

        files_dir = dm.datasets_path / "ds1" / "files"
        assert (files_dir / "a.pdf").read_bytes() == b"%PDF-fake"
        assert (files_dir / "sub" / "b.txt").read_text() == "hello"

        meta = json.loads((dm.datasets_path / "ds1" / "meta.json").read_text())
        assert meta["name"] == "ds1"
        assert meta["description"] == "test export"
        assert meta["document_count"] == 0
        assert "password_hash" not in meta


def test_prepare_import_strips_fingerprint_and_counts():
    with tempfile.TemporaryDirectory() as td:
        dm, _ = _make_dm(Path(td))
        tar = Path(td) / "backup.tar.gz"
        _build_export(tar, name="ds1", files={"a.txt": b"x"})
        # (meta.json written by _build_export carries no fingerprint fields —
        # add them to prove they are stripped.)
        import tarfile as tf

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            meta = {
                "name": "ds1",
                "embedder_model": "old-embedder",
                "embedder_dim": 123,
                "file_type_counts": {"pdf": 9},
                "document_count": 9,
            }
            info = tf.TarInfo("meta.json")
            payload = json.dumps(meta).encode()
            info.size = len(payload)
            t.addfile(info, io.BytesIO(payload))
            info = tf.TarInfo("files/a.txt")
            info.size = 1
            t.addfile(info, io.BytesIO(b"x"))
        tar.write_bytes(buf.getvalue())

        dm.prepare_import(tar)
        meta = json.loads((dm.datasets_path / "ds1" / "meta.json").read_text())
        for stale in ("embedder_model", "embedder_dim", "file_type_counts"):
            assert stale not in meta, f"{stale} must be stripped so the current embedder re-stamps"


def test_import_dataset_files_mode_calls_recreate():
    with tempfile.TemporaryDirectory() as td:
        dm, recorded = _make_dm(Path(td))
        tar = Path(td) / "backup.tar.gz"
        _build_export(tar, name="ds1", files={"a.txt": b"x"})
        result = dm.import_dataset(tar)
        assert result["mode"] == "re-embed-files"
        assert recorded["recreated"] == ["ds1"]


# ---------------------------------------------------------------------------
# Name collisions
# ---------------------------------------------------------------------------


def test_import_collision_raises_without_overwrite():
    with tempfile.TemporaryDirectory() as td:
        dm, recorded = _make_dm(Path(td))
        (dm.datasets_path / "ds1").mkdir()
        tar = Path(td) / "backup.tar.gz"
        _build_export(tar, name="ds1", files={"a.txt": b"x"})
        try:
            dm.prepare_import(tar)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        assert recorded["deleted"] == []


def test_import_overwrite_deletes_existing():
    with tempfile.TemporaryDirectory() as td:
        dm, recorded = _make_dm(Path(td))
        (dm.datasets_path / "ds1").mkdir()
        tar = Path(td) / "backup.tar.gz"
        _build_export(tar, name="ds1", files={"a.txt": b"x"})
        dm.prepare_import(tar, overwrite=True)
        assert recorded["deleted"] == ["ds1"]


def test_import_new_name_and_traversal_members():
    with tempfile.TemporaryDirectory() as td:
        dm, _ = _make_dm(Path(td))
        tar = Path(td) / "backup.tar.gz"
        _build_export(tar, name="ds1", files={"a.txt": b"x"})
        plan = dm.prepare_import(tar, new_name="restored-ds")
        assert plan["dataset"] == "restored-ds"
        assert (dm.datasets_path / "restored-ds" / "meta.json").exists()

        # A member escaping the export layout must be rejected outright.
        evil = Path(td) / "evil.tar.gz"
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            info = tarfile.TarInfo("../evil.txt")
            info.size = 2
            t.addfile(info, io.BytesIO(b"no"))
        evil.write_bytes(buf.getvalue())
        try:
            dm.prepare_import(evil)
            raise AssertionError("expected ValueError for traversal member")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# text-replay mode
# ---------------------------------------------------------------------------

ROWS = [
    {"id": "1", "payload": {"page_content": "real note one", "metadata": {"source": "/old/ds/file.py", "page": 1}}},
    {
        "id": "2",
        "payload": {
            "page_content": "[Image: photo.jpg]",
            "metadata": {"source": "/old/ds/photo.jpg", "image": "data:image/jpeg;base64,AA=="},
        },
    },
    {
        "id": "3",
        "payload": {
            "page_content": "real note two",
            "metadata": {"source": "/old/ds/x.log", "severities": ["ERROR"], "image": "data:image/png;base64,AA=="},
        },
    },
]


def test_import_dataset_text_replay():
    with tempfile.TemporaryDirectory() as td:
        dm, recorded = _make_dm(Path(td))
        tar = Path(td) / "backup.tar.gz"
        _build_export(tar, name="rawds", rows=ROWS)

        result = dm.import_dataset(tar)
        assert result["mode"] == "text-replay"

        added = recorded["added"]
        assert len(added) >= 1
        docs = [d for batch in added for d in batch]
        texts = [d["text"] for d in docs]
        assert texts == ["real note one", "real note two"], "placeholder rows are skipped"
        two = next(d for d in docs if d["text"] == "real note two")
        assert "image" not in two, "media payload keys must be stripped in replay mode"
        assert two["severities"] == ["ERROR"], "lightweight metadata is preserved"
        assert two["source"] == "/old/ds/x.log"


def test_import_restores_password():
    with tempfile.TemporaryDirectory() as td:
        dm, _ = _make_dm(Path(td))
        tar = Path(td) / "backup.tar.gz"
        _build_export(tar, name="ds1", files={"a.txt": b"x"})
        dm.import_dataset(tar, password="s3cret")
        meta = json.loads((dm.datasets_path / "ds1" / "meta.json").read_text())
        assert meta.get("password_hash"), "password must be re-applied on restore"


def test_import_empty_backup_raises():
    with tempfile.TemporaryDirectory() as td:
        dm, recorded = _make_dm(Path(td))
        tar = Path(td) / "backup.tar.gz"
        _build_export(tar, name="ds1", files={}, rows=[])
        try:
            dm.prepare_import(tar)
            raise AssertionError("expected ValueError for empty backup")
        except ValueError as exc:
            assert "nothing to restore" in str(exc)
        assert recorded["deleted"] == ["ds1"], "half-made dataset is cleaned up"


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  {fn.__name__} ... OK")
        except Exception:
            failed += 1
            print(f"  {fn.__name__} ... FAIL")
            traceback.print_exc()
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) failed'}")
    sys.exit(1 if failed else 0)
