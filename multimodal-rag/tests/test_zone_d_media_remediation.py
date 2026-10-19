"""Zone D remediation tests — media pipeline (audit 2026-10-16).

Pins the three fixes from the consolidated audit (``documentation/AUDIT-2026-10-16.md``),
all offline (no model endpoints, no Qdrant server):

* **P1-5** — the embedder http(s) media fetch was
  ``http_async_client.get(url, follow_redirects=True)`` + ``response.content``: the whole
  body buffered into RAM with no ceiling, and every redirect hop followed blind.
  ``MAX_MEDIA_FETCH_BYTES`` / per-hop re-validation existed only in
  ``rag_system._afetch_media_bytes``.  Now both ``_fetch_obj_async`` and
  ``_fetch_video_frames`` go through ``_afetch_media_bounded``.
* **P2 segment clobber** — ``VideoProcessor.process_iter`` took the transcode-failure
  branch WITHOUT advancing the index that names ``seg{N}.mp4``, so the next successful
  window overwrote the file written for the previous successful window; the loss was also
  never summarised.  The write index now tracks the WINDOW, and failures are logged.
* **P2 empty tier-3 payload** — ``_resize_media_in_docs`` b64-encoded ``proc.stdout``
  unconditionally, so a failed ffmpeg stored ``data:video/mp4;base64,`` (empty) as the
  tier-3 payload the reranker/VLM consume directly.  Failures are now skipped.

The tests use a FAKE transport (a real ``httpx2`` client with a ``MockTransport``), so the
whole request path — build_request, send(stream=True), aiter_bytes, redirect handling — is
exercised for real; only the socket is stubbed.  Every test is timeout-safe: no DNS, no
network, no subprocess.

Run::

    pytest tests/test_zone_d_media_remediation.py -q
"""

import asyncio
import base64
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import multimodal_rag.utils.media_paths as mp
from multimodal_rag.utils import model_adapters as ma
from multimodal_rag.utils.model_adapters import InputConversion

_JPEG = b"\xff\xd8\xff\xe0" + b"JFIF-payload" * 8


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def media_env(tmp_path, monkeypatch):
    """Allowlist pointed at tmp so the file:// branch stays usable.

    media_paths freezes its prefix tuple at import, so patch the module
    attributes (the documented house pattern from test_media_fetch_allowlist.py).
    """
    datasets = tmp_path / "datasets"
    staging = tmp_path / "staging"
    datasets.mkdir()
    staging.mkdir()
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    monkeypatch.delenv("MEDIA_ALLOW_PATH_PREFIXES", raising=False)
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_PATH_PREFIXES", (str(datasets), str(staging)))
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_ANY", False)
    return tmp_path


@pytest.fixture(autouse=True)
def _offline_dns(monkeypatch):
    """Make every test hostname resolve to one public IP — and NOTHING else.

    The fetch policy resolves DNS at validate time (that is the DNS-rebinding
    pin).  The test hostnames are not real, so without this they would resolve
    to nothing and be refused as suspicious, masking what each test asserts.
    The stub also proves the tests never touch the network: any getaddrinfo
    call is answered locally from this table.

    Literal IPs are classified WITHOUT resolution, so the private-redirect
    test still exercises the real private-range branch.
    """
    import socket

    table = {
        "media.example": "93.184.216.34",
        "cdn.example": "93.184.216.35",
        "other.example": "93.184.216.36",
    }

    def fake_getaddrinfo(host, port, *args, **kwargs):
        ip = table.get(str(host).lower())
        if ip is None:
            raise socket.gaierror(f"unresolvable in tests: {host}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def _conv(client=None, max_pixels=0) -> InputConversion:
    """InputConversion over a stub embedder wired to *client* (default: none)."""
    emb = SimpleNamespace(mm_processor_kwargs={"max_pixels": max_pixels}, http_async_client=client)
    return InputConversion(emb)


def _stream_client(handler) -> httpx2.AsyncClient:
    """A real httpx2.AsyncClient whose transport is a MockTransport.

    ``follow_redirects`` is off on the client by design: the fix follows hops
    manually, so the test must not let the transport paper over a missed hop.
    """
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler), follow_redirects=False)


def _body_of(n: int) -> bytes:
    return b"x" * n


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# P1-5 (a) — the size cap is ENFORCED on both fetch sites
# ===========================================================================


def test_obj_fetch_refuses_body_over_cap(monkeypatch):
    """A response larger than MAX_MEDIA_FETCH_BYTES is refused, not buffered."""
    monkeypatch.setattr(ma, "_MAX_MEDIA_FETCH_BYTES", 1024)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "image/jpeg"}, content=_body_of(4096))

    conv = _conv(_stream_client(handler))
    with pytest.raises(ValueError, match="exceeds MAX_MEDIA_FETCH_BYTES"):
        _run(conv._fetch_obj_async("http://media.example/a.jpg"))


def test_video_frames_fetch_refuses_body_over_cap(monkeypatch):
    """The video-frames path is capped identically (the worse leak: those
    bytes went on into PyAV/ffmpeg)."""
    monkeypatch.setattr(ma, "_MAX_MEDIA_FETCH_BYTES", 1024)
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        return httpx2.Response(200, headers={"content-type": "video/mp4"}, content=_body_of(8192))

    conv = _conv(_stream_client(handler))
    with pytest.raises(ValueError, match="exceeds MAX_MEDIA_FETCH_BYTES"):
        _run(conv._fetch_video_frames("http://media.example/v.mp4", num_frames=2))
    # Refused at the fetch: nothing was decoded (no second call, no ffmpeg).
    assert calls["n"] == 1


def test_cap_boundary_exactly_at_limit_is_allowed(monkeypatch):
    """The cap is a ceiling, not an exclusive bound: len == cap passes."""
    monkeypatch.setattr(ma, "_MAX_MEDIA_FETCH_BYTES", 512)
    payload = _body_of(512)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "image/jpeg"}, content=payload)

    b64, mime = _run(_conv(_stream_client(handler))._fetch_obj_async("http://media.example/ok.jpg"))
    assert base64.b64decode(b64) == payload
    assert mime == "image/jpeg"


def test_cap_disabled_by_zero(monkeypatch):
    """MAX_MEDIA_FETCH_BYTES=0 disables the ceiling (documented escape)."""
    monkeypatch.setattr(ma, "_MAX_MEDIA_FETCH_BYTES", 0)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "image/jpeg"}, content=_body_of(1 << 20))

    b64, _ = _run(_conv(_stream_client(handler))._fetch_obj_async("http://media.example/big.jpg"))
    assert len(base64.b64decode(b64)) == 1 << 20


def test_cap_reads_the_same_env_var_as_the_guarded_path(monkeypatch):
    """One env contract for both fetch sites: MAX_MEDIA_FETCH_BYTES.

    Re-importing the module with the env set must pick the value up — the
    guarded path (rag_system) reads the same name at import, and the two must
    never disagree.
    """
    import importlib

    monkeypatch.setenv("MAX_MEDIA_FETCH_BYTES", "1234")
    try:
        reloaded = importlib.reload(ma)
        assert reloaded._MAX_MEDIA_FETCH_BYTES == 1234
    finally:
        monkeypatch.delenv("MAX_MEDIA_FETCH_BYTES", raising=False)
        importlib.reload(ma)
    assert ma._MAX_MEDIA_FETCH_BYTES == 512 * 1024 * 1024


# ===========================================================================
# P1-5 (b) — per-hop redirect re-validation
# ===========================================================================


def test_redirect_hops_are_revalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A redirect to a BLOCKED host is refused at the hop, not followed blind.

    This is the SSRF half of P1-5: with follow_redirects=True the policy saw
    only the original URL, so a public URL could bounce the fetch into private
    space.  The stub transport returns a 302 to a literal private IP; the
    per-hop validate_fetch_url must reject it before any second request.
    """
    monkeypatch.setenv("INGEST_BLOCK_PRIVATE_HOSTS", "true")
    monkeypatch.delenv("INGEST_ALLOW_HOSTS", raising=False)
    import importlib

    from multimodal_rag.utils import url_policy as up

    importlib.reload(up)
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(str(request.url))
        return httpx2.Response(302, headers={"location": "http://10.9.9.9/secret.jpg"})

    conv = _conv(_stream_client(handler))
    try:
        with pytest.raises(ValueError, match="private/internal"):
            _run(conv._fetch_obj_async("http://media.example/redirect.jpg"))
    finally:
        importlib.reload(up)
    # Exactly one request: the redirect target was refused before being fetched.
    assert len(seen) == 1


def test_redirect_hop_without_location_is_refused(monkeypatch):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(302)  # no Location header

    with pytest.raises(ValueError, match="no Location header"):
        _run(_conv(_stream_client(handler))._fetch_obj_async("http://media.example/x.jpg"))


def test_redirect_limit_is_enforced(monkeypatch):
    """A redirect loop terminates at _MEDIA_MAX_REDIRECTS instead of spinning."""
    monkeypatch.setattr(ma, "_MEDIA_MAX_REDIRECTS", 2)
    hops = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        hops["n"] += 1
        return httpx2.Response(302, headers={"location": "http://media.example/loop.jpg"})

    with pytest.raises(ValueError, match="exceeded 2 redirects"):
        _run(_conv(_stream_client(handler))._fetch_obj_async("http://media.example/loop.jpg"))
    assert hops["n"] == 3  # the initial request + the 2 allowed hops


def test_redirect_to_public_host_is_followed(monkeypatch):
    """The happy path survives: a permitted hop is followed and its body used."""
    monkeypatch.delenv("INGEST_ALLOW_HOSTS", raising=False)
    payload = _body_of(64)

    def handler(request: httpx2.Request) -> httpx2.Response:
        if str(request.url).endswith("/final.jpg"):
            return httpx2.Response(200, headers={"content-type": "image/jpeg"}, content=payload)
        return httpx2.Response(302, headers={"location": "http://cdn.example/final.jpg"})

    b64, mime = _run(_conv(_stream_client(handler))._fetch_obj_async("http://media.example/start.jpg"))
    assert base64.b64decode(b64) == payload
    # mime comes from the FINAL hop, not the redirecting one.
    assert mime == "image/jpeg"


def test_mime_type_comes_from_the_final_hop(monkeypatch):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if str(request.url).endswith("/final"):
            return httpx2.Response(200, headers={"content-type": "video/mp4; charset=x"}, content=b"v")
        return httpx2.Response(302, headers={"location": "http://cdn.example/final"})

    _, mime = _run(_conv(_stream_client(handler))._fetch_obj_async("http://media.example/a"))
    assert mime == "video/mp4"  # charset stripped, final hop used


def test_file_branch_is_untouched(media_env):
    """The file:// path keeps its previous behaviour (no HTTP, no cap)."""
    img = media_env / "datasets" / "ds" / "files" / "a.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(_JPEG)
    b64, mime = _run(_conv()._fetch_obj_async(str(img)))
    assert base64.b64decode(b64) == _JPEG
    assert mime == "image/jpeg"


# ===========================================================================
# P2 — video segment clobber + silent loss
# ===========================================================================


def _video_processor(**kw):
    from multimodal_rag.input_processing.video_processor import VideoProcessor

    return VideoProcessor(**kw)


def test_failed_segment_does_not_clobber_a_previous_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """THE regression: a failure must not make the next success reuse its index.

    Windows: 0 ok, 1 FAIL, 2 ok.  The old code named files by the success
    counter, so window 2 wrote ``seg1.mp4`` — the very file window 0 had
    already written (clobber), and the failure was invisible.  Now window 2
    writes ``seg2.mp4`` and window 1 leaves no file.
    """
    vp = _video_processor(segment_seconds=10, overlap_seconds=0, fps=1.0)
    monkeypatch.setattr(
        type(vp), "_probe_metadata", staticmethod(lambda p: {"duration": 25.0, "width": 640, "height": 480})
    )

    seen_starts: list[float] = []

    def fake_transcode(input_path: Any, start: float, duration: float, fps: float, scale_filter: Any) -> bytes:
        seen_starts.append(start)
        # Window at t=10 fails; every other window succeeds with a UNIQUE body.
        if abs(start - 10.0) < 0.01:
            return b""
        return f"SEG-{start:.0f}".encode()

    monkeypatch.setattr(type(vp), "_transcode_segment", staticmethod(fake_transcode))

    docs = list(vp.process_iter("movie.mp4", save_dir=str(tmp_path)))

    # Two documents (windows 0 and 2); window 1 produced nothing.
    assert [d["timestamp_start"] for d in docs] == [0.0, 20.0]
    assert seen_starts == [0.0, 10.0, 20.0]

    files = sorted(p.name for p in tmp_path.glob("*.mp4"))
    # Window 0 -> seg0.mp4, window 1 -> nothing, window 2 -> seg2.mp4.
    assert files == ["movie_seg0.mp4", "movie_seg2.mp4"], files
    # No file was overwritten: each surviving file holds that window body.
    assert (tmp_path / "movie_seg0.mp4").read_bytes() == b"SEG-0"
    assert (tmp_path / "movie_seg2.mp4").read_bytes() == b"SEG-20"
    # seg1.mp4 must NOT exist — the failed window leaves no gap-filler.
    assert not (tmp_path / "movie_seg1.mp4").exists()


def test_failed_segment_is_logged_not_silent(tmp_path, monkeypatch, caplog):
    """A partially-transcoded video reports the loss instead of going quiet."""
    import logging

    vp = _video_processor(segment_seconds=10, overlap_seconds=0, fps=1.0)
    monkeypatch.setattr(
        type(vp), "_probe_metadata", staticmethod(lambda p: {"duration": 20.0, "width": 640, "height": 480})
    )
    monkeypatch.setattr(type(vp), "_transcode_segment", staticmethod(lambda *a, **k: b""))

    with caplog.at_level(logging.WARNING):
        docs = list(vp.process_iter("broken.mp4", save_dir=str(tmp_path)))

    assert docs == []
    text = caplog.text
    assert "failed to transcode" in text
    # The summary names the count of lost windows.
    assert "2 of 2 window(s) failed" in text
    assert not list(tmp_path.glob("*.mp4"))


def test_all_windows_succeed_keeps_contiguous_numbering(tmp_path, monkeypatch):
    """The happy path is unchanged: contiguous seg0..segN, one doc each."""
    vp = _video_processor(segment_seconds=10, overlap_seconds=0, fps=1.0)
    monkeypatch.setattr(
        type(vp), "_probe_metadata", staticmethod(lambda p: {"duration": 30.0, "width": 640, "height": 480})
    )
    monkeypatch.setattr(type(vp), "_transcode_segment", staticmethod(lambda i, s, d, f, sc: f"SEG-{s:.0f}".encode()))

    docs = list(vp.process_iter("ok.mp4", save_dir=str(tmp_path)))

    assert len(docs) == 3
    assert sorted(p.name for p in tmp_path.glob("*.mp4")) == ["ok_seg0.mp4", "ok_seg1.mp4", "ok_seg2.mp4"]


def test_failed_segment_keeps_the_overlap_break(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trailing-overlap break still keys off the WINDOW, not the success count.

    With a failure run the old code kept seg_count at 0, so the
    ``remaining < overlap`` guard never fired and the loop emitted extra
    tail windows.  The window counter restores the original stop condition.
    """
    vp = _video_processor(segment_seconds=10, overlap_seconds=4, fps=1.0)
    monkeypatch.setattr(
        type(vp), "_probe_metadata", staticmethod(lambda p: {"duration": 12.0, "width": 640, "height": 480})
    )
    starts: list[float] = []

    def fake(input_path: Any, start: float, duration: float, fps: float, scale_filter: Any) -> bytes:
        starts.append(start)
        return b""

    monkeypatch.setattr(type(vp), "_transcode_segment", staticmethod(fake))
    list(vp.process_iter("short.mp4"))

    # window t=0 (remaining 12 >= overlap 4), then t=6 (remaining 6 >= 4);
    # t=12 ends the while loop (t_start >= duration).  No extra tail window.
    assert starts == [0.0, 6.0]


# ===========================================================================
# P2 — empty tier-3 video payload
# ===========================================================================


def _rag_for_resize():
    """A MultimodalRAG whose embedder stub has max_pixels>0 (resize active).

    ``embed`` is a ``cached_property`` that would build a real model client from
    the stub, so seed the instance attribute directly (cached_property stores in
    ``__dict__``, so an explicit assignment wins).  It is a real
    ``MultiModalEmbeddings`` — the image branch calls
    ``self.embed.input_conversion._resize_image``.
    """
    from multimodal_rag.rag_system import MultimodalRAG
    from multimodal_rag.utils.model_adapters import MultiModalEmbeddings

    emb = SimpleNamespace(
        mm_processor_kwargs={"max_pixels": 720 * 720, "fps": 1.0},
        chunk_size=64,
        chunk_overlap=8,
        text_splitter=None,
        allowable_modalities=("text", "image", "video"),
    )
    rag = MultimodalRAG.__new__(MultimodalRAG)
    rag.embedder = emb
    rag.embed = MultiModalEmbeddings(emb)
    return rag


def _stub_ffmpeg(monkeypatch, returncode, stdout):
    """Patch the subprocess calls used by _resize_media_in_docs.

    The original ``subprocess.run`` is captured BEFORE patching (delegating to
    ``real_sp.run`` after the patch would recurse into the stub forever).
    """
    import subprocess as real_sp

    original_run = real_sp.run

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "ffmpeg":
            return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=b"boom")
        if cmd and cmd[0] == "ffprobe":
            return SimpleNamespace(
                returncode=0,
                stdout=b'{"streams": [{"width": 1920, "height": 1080}]}',
                stderr=b"",
            )
        return original_run(cmd, **kwargs)

    monkeypatch.setattr(real_sp, "run", fake_run)


def test_failed_ffmpeg_stores_no_empty_video_payload(tmp_path, monkeypatch, media_env):
    """THE regression: a failed transcode must not become data:video/mp4;base64,"""
    rag = _rag_for_resize()
    _stub_ffmpeg(monkeypatch, returncode=1, stdout=b"")

    src = media_env / "datasets" / "clip.mp4"
    src.write_bytes(b"not-really-a-video")
    docs = [{"text": "[Video: clip.mp4]", "video": str(src), "source": str(src)}]

    out = rag._resize_media_in_docs(docs)

    assert len(out) == 1
    # The empty payload is gone entirely — the key is dropped, not blanked.
    assert "video" not in out[0], out[0]
    assert not any(
        isinstance(v, str) and v.startswith("data:video/") and v.endswith("base64,") for v in out[0].values()
    )


def test_successful_ffmpeg_still_stores_the_video_payload(tmp_path, monkeypatch, media_env):
    """Positive control: a good transcode keeps its tier-3 data URL."""
    rag = _rag_for_resize()
    _stub_ffmpeg(monkeypatch, returncode=0, stdout=b"REAL-MP4-BYTES")

    src = media_env / "datasets" / "clip.mp4"
    src.write_bytes(b"not-really-a-video")
    docs = [{"text": "[Video: clip.mp4]", "video": str(src), "source": str(src)}]

    out = rag._resize_media_in_docs(docs)

    val = out[0]["video"]
    assert val.startswith("data:video/mp4;base64,")
    assert base64.b64decode(val.split(",", 1)[1]) == b"REAL-MP4-BYTES"


def test_empty_stdout_with_zero_exit_is_also_refused(tmp_path, monkeypatch, media_env):
    """Exit 0 with no bytes is still a failed payload (belt and braces)."""
    rag = _rag_for_resize()
    _stub_ffmpeg(monkeypatch, returncode=0, stdout=b"")

    src = media_env / "datasets" / "clip.mp4"
    src.write_bytes(b"x")
    out = rag._resize_media_in_docs([{"text": "t", "video": str(src)}])
    assert "video" not in out[0]


def test_failed_video_leaves_a_sibling_image_intact(tmp_path, monkeypatch, media_env):
    """A skipped video must not disturb other media keys in the same doc."""
    from PIL import Image

    rag = _rag_for_resize()
    _stub_ffmpeg(monkeypatch, returncode=1, stdout=b"")

    # The bare-path read site enforces the media allowlist (defence in depth),
    # so the fixture must live inside the prefixes.
    img = media_env / "datasets" / "big.jpg"
    Image.new("RGB", (4000, 4000), (10, 20, 30)).save(img, format="JPEG")
    vid = media_env / "datasets" / "clip.mp4"
    vid.write_bytes(b"x")

    out = rag._resize_media_in_docs([{"text": "t", "image": str(img), "video": str(vid)}])
    assert "video" not in out[0]
    assert out[0]["image"].startswith("data:image/")


def test_max_pixels_zero_short_circuits_untouched():
    """The documented early return still leaves docs alone."""
    from multimodal_rag.rag_system import MultimodalRAG

    rag = MultimodalRAG.__new__(MultimodalRAG)
    rag.embedder = SimpleNamespace(mm_processor_kwargs={"max_pixels": 0}, chunk_size=64)
    docs = [{"text": "t", "video": "/nope/x.mp4"}]
    assert rag._resize_media_in_docs(docs) == docs
