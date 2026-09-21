"""Regression tests for the preprocessed-media corruption fix.

Root cause (diagnosed 2026-09-21): ``_preprocess_video_file`` (and the image
/ audio siblings) wrote ffmpeg's output DIRECTLY onto the final shared
``*_preprocessed`` path.  Two writers — e.g. two of the four
``rag-mcp-server`` replicas preprocessing the same dedup'd upload on the
shared PVC — interleave their muxer finalizations and leave an unplayable
file: duplicated ``moov`` + ``mdat`` overrunning EOF.  Browsers then refuse
to play ("the video can't be played because the file is corrupt") while the
search index is fine.

The fix: write to a unique temp sibling, validate (ffprobe + MP4 atom walk),
then atomically rename; the writer also skips work when a valid sibling
already exists (idempotency).

Run::

    pytest tests/full_pipeline/test_preprocess_atomicity.py
    python tests/full_pipeline/test_preprocess_atomicity.py   # standalone
"""

import os
import subprocess as sp
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.dataset_manager import (
    DatasetManager,
    _media_file_ok,
    _mp4_boxes_broken,
    _preprocess_image_file,
    _preprocess_video_file,
)

FFMPEG = "ffmpeg"


def _make_manager(tmpdir: Path) -> DatasetManager:
    dm = DatasetManager.__new__(DatasetManager)
    dm.base_path = tmpdir
    dm.datasets_path = tmpdir / "datasets"
    dm.datasets_path.mkdir(parents=True, exist_ok=True)
    files = dm.datasets_path / "ds" / "files"
    files.mkdir(parents=True, exist_ok=True)
    return dm


def _make_test_video(path: Path, seconds: int = 2, fps: int = 30, scale: bool = True) -> None:
    """A small but real H.264/AAC MP4 (the race needs a real muxer)."""
    cmd = [
        FFMPEG, "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate={fps}",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-t", str(seconds),
    ]
    if scale:
        cmd += ["-vf", "scale=640:480"]
    cmd += [
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-y", str(path),
    ]
    r = sp.run(cmd, capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")


def _boxes(path: Path) -> list[tuple[str, int]]:
    data = path.read_bytes()
    n = len(data)
    off = 0
    out = []
    while off + 8 <= n:
        sz = int.from_bytes(data[off : off + 4], "big")
        typ = data[off + 4 : off + 8].decode("latin-1", "replace")
        if sz == 1:
            sz = int.from_bytes(data[off + 8 : off + 16], "big")
        elif sz == 0:
            sz = n - off
        if sz < 8 or off + sz > n:
            out.append((typ, sz))
            break
        out.append((typ, sz))
        off += sz
    return out


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def test_mp4_validator_catches_truncated_mdat():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v.mp4"
        _make_test_video(p, scale=False)
        good = p.read_bytes()
        assert not _mp4_boxes_broken(p)
        p.write_bytes(good[:-1000])  # chop the tail
        assert _mp4_boxes_broken(p), "truncated mdat must be flagged"
        assert not _media_file_ok(p)


def test_mp4_validator_catches_duplicate_moov():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v.mp4"
        _make_test_video(p, scale=False)
        data = p.read_bytes()
        ftyp, rest = data[:32], data[32:]
        p.write_bytes(ftyp + rest + rest)  # splice a second trailer
        assert _mp4_boxes_broken(p), "duplicated moov must be flagged"
        assert not _media_file_ok(p)


def test_mp4_validator_accepts_healthy_file():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v.mp4"
        _make_test_video(p)
        assert _media_file_ok(p), "a normal faststart MP4 must pass"


# ---------------------------------------------------------------------------
# The race itself
# ---------------------------------------------------------------------------


def test_concurrent_preprocess_writers_leave_valid_output():
    """Two transcodes racing on one output path must end with a VALID file.

    This is the exact production failure: two replicas preprocessing the same
    source concurrently.  With the pre-fix direct-to-final-path writer this
    deterministically produced moov/moov/free/mdat-overrun.
    """
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src = d / "abc123_src.mp4"
        _make_test_video(src, seconds=4, scale=False)  # big enough that runs overlap

        results: list[Path] = []
        errors: list[str] = []

        def run(tag: str) -> None:
            try:
                results.append(_preprocess_video_file(src, max_pixels=320 * 240, max_fps=15))
            except Exception as exc:  # pragma: no cover
                errors.append(f"{tag}: {exc}")

        threads = [threading.Thread(target=run, args=(t,)) for t in ("A", "B", "C")]
        threads[0].start()
        threads[1].start()
        threads[2].start()
        for t in threads:
            t.join()

        assert not errors, errors
        out = d / "abc123_src_preprocessed.mp4"
        assert out.exists(), "preprocessed sibling must exist"
        assert _media_file_ok(out), (
            f"concurrent writers left a broken file: boxes={_boxes(out)}"
        )
        # No leftover temp siblings
        leftovers = [p.name for p in d.iterdir() if ".tmp" in p.name]
        assert not leftovers, f"temp files leaked: {leftovers}"


def test_preprocess_skips_when_valid_sibling_exists():
    """Idempotency: a valid preprocessed sibling is never clobbered."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src = d / "abc123_src.mp4"
        _make_test_video(src, scale=False)
        pre = d / "abc123_src_preprocessed.mp4"
        _make_test_video(pre, seconds=1, scale=False)  # valid but different
        before = pre.stat().st_mtime_ns
        result = _preprocess_video_file(src, max_pixels=320 * 240, max_fps=15)
        assert result == pre
        assert pre.stat().st_mtime_ns == before, "valid sibling must not be rewritten"


def test_broken_sibling_is_regenerated():
    """A corrupt preprocessed sibling (the production damage) gets replaced."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src = d / "abc123_src.mp4"
        _make_test_video(src, scale=False)
        pre = d / "abc123_src_preprocessed.mp4"
        _make_test_video(pre, scale=False)
        data = pre.read_bytes()
        pre.write_bytes(data[:-500])  # simulate the truncation damage
        assert not _media_file_ok(pre)

        result = _preprocess_video_file(src, max_pixels=320 * 240, max_fps=15)
        assert result == pre
        assert _media_file_ok(pre), "broken sibling must be regenerated"


def test_failing_transcode_keeps_original_and_no_garbage():
    """If validation fails, the writer must not promote a broken file."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src = d / "abc123_src.mp4"
        _make_test_video(src, scale=False)
        # Point "ffmpeg" at something that always fails to produce output.
        broken_dir = d / "fakebin"
        broken_dir.mkdir()
        (broken_dir / "ffmpeg").write_text("#!/bin/sh\nexit 1\n")
        (broken_dir / "ffmpeg").chmod(0o755)

        env_path = os.environ["PATH"]

        class _Raiser:
            def __enter__(self):
                os.environ["PATH"] = str(broken_dir) + os.pathsep + env_path
                return self

            def __exit__(self, *a):
                os.environ["PATH"] = env_path
                return False

        with _Raiser():
            result = _preprocess_video_file(src, max_pixels=320 * 240, max_fps=15)
        assert result == src, "failed transcode must return the original"
        assert not (d / "abc123_src_preprocessed.mp4").exists(), "no invalid output may be promoted"
        leftovers = [p.name for p in d.iterdir() if ".tmp" in p.name]
        assert not leftovers, f"temp files leaked: {leftovers}"


# ---------------------------------------------------------------------------
# Image writer + repair routine
# ---------------------------------------------------------------------------


def test_image_preprocess_atomic_and_validated():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src = d / "abc123_img.jpg"
        # A real 2000x1500 JPEG (exceeds the default cap) built via PIL
        from PIL import Image

        Image.new("RGB", (2000, 1500), (180, 40, 40)).save(src, "JPEG")
        result = _preprocess_image_file(src, max_pixels=1000 * 1000)
        pre = d / "abc123_img_preprocessed.jpg"
        assert result == pre
        assert pre.exists()
        with Image.open(pre) as img:
            assert img.size[0] * img.size[1] <= 1000 * 1000


def test_repair_preprocessed_media_repairs_broken_video():
    with tempfile.TemporaryDirectory() as td:
        dm = _make_manager(Path(td))
        files = dm.datasets_path / "ds" / "files"
        orig = files / "abc123_vid.mp4"
        _make_test_video(orig, scale=False)
        pre = files / "abc123_vid_preprocessed.mp4"
        _make_test_video(pre, scale=False)
        pre.write_bytes(pre.read_bytes()[:-500])  # the production damage

        report = dm.repair_preprocessed_media("ds", dry_run=True)
        assert report["checked"] == 1
        assert report["repaired"] and report["repaired"][0]["action"] == "would-regenerate"
        assert pre.exists(), "dry run must not touch the file"

        report = dm.repair_preprocessed_media("ds", dry_run=False)
        assert report["checked"] == 1
        assert len(report["repaired"]) == 1
        assert not report["failed"]
        assert _media_file_ok(pre), "repaired file must be valid"


def test_repair_leaves_valid_files_alone():
    with tempfile.TemporaryDirectory() as td:
        dm = _make_manager(Path(td))
        files = dm.datasets_path / "ds" / "files"
        orig = files / "abc123_vid.mp4"
        _make_test_video(orig, scale=False)
        pre = files / "abc123_vid_preprocessed.mp4"
        _make_test_video(pre, scale=False)
        before = pre.read_bytes()

        report = dm.repair_preprocessed_media("ds")
        assert report["checked"] == 1
        assert report["repaired"] == [] and report["failed"] == []
        assert pre.read_bytes() == before, "valid file must not be touched"


def test_repair_reports_missing_original():
    with tempfile.TemporaryDirectory() as td:
        dm = _make_manager(Path(td))
        files = dm.datasets_path / "ds" / "files"
        # A preprocessed file whose tier-1 original is gone — not repairable.
        orphan = files / "abc123_vid_preprocessed.mp4"
        orphan.write_bytes(b"\x00\x00\x00\x08free" * 4)

        report = dm.repair_preprocessed_media("ds")
        assert report["checked"] == 0, "no original → not a repair candidate"


def test_webm_is_never_transcoded():
    """A .webm original must not be transcoded — the libx264/aac output
    cannot go into a WebM container, so every attempt was doomed: ffmpeg
    wrote a tiny container stub that the old size>0 check promoted as
    "preprocessed" (the 'Probably the Best Chicken Dish' corruption)."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src = d / "abc123_src.webm"
        # A real VP9/Opus webm (any webm is skipped by suffix, but make it
        # oversized so a transcode would also be attempted-and-slow).
        sp.run(
            [
                FFMPEG, "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10",
                "-t", "1",
                "-c:v", "libvpx-vp9", "-b:v", "200k",
                "-c:a", "libopus",
                "-y", str(src),
            ],
            capture_output=True,
        )
        if not src.exists():  # libvpx/opus encoders unavailable in test env
            src.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 4096)
        # An original whose pixels/fps are within limits: preprocess must
        # return the ORIGINAL (no sibling at all).
        result = _preprocess_video_file(src, max_pixels=1000 * 1000, max_fps=24)
        assert result == src, "webm must be returned unchanged"
        assert not (d / "abc123_src_preprocessed.webm").exists(), "no doomed transcode may be attempted"


def test_repair_reports_webm_stub_as_expected_failure():
    """A corrupt webm stub is deletable but not regenerable: repair reports
    it in `failed` with the container reason instead of silently succeeding."""
    with tempfile.TemporaryDirectory() as td:
        dm = _make_manager(Path(td))
        files = dm.datasets_path / "ds" / "files"
        orig = files / "abc123_vid.webm"
        orig.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 4096)
        stub = files / "abc123_vid_preprocessed.webm"
        stub.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 64)  # the stub damage

        report = dm.repair_preprocessed_media("ds")
        assert report["checked"] == 1
        assert report["repaired"] == []
        assert len(report["failed"]) == 1
        assert "container" in report["failed"][0]["reason"]
        # The broken stub was removed; the original stays untouched.
        assert not stub.exists()
        assert orig.exists()


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
