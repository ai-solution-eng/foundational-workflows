"""Embedder media-fetch allowlist tests (audit 2026-10-02, P0-2/P1-3).

The audit's P0-2 finding: the embedder's media-fetch path opened arbitrary
local files.  ``InputConversion._fetch_obj_async`` (and its sync twin
``_add_raw_url``, and ``_fetch_video_frames``) did
``url.removeprefix("file://")`` → ``open(url, "rb")`` with NO allowlist
check, and ``url_policy._check_media_url_policy`` returned IMMEDIATELY for
non-http(s) URLs — so ``file:///etc/passwd`` passed every gate that funnels
through the policy and came back as base64 in the embed request.

The fix: every fetch site in ``utils/model_adapters.py`` runs
``_guard_media_fetch`` before touching I/O, and ``_check_media_url_policy``
now enforces the SAME ``media_paths._media_path_allowed`` allowlist
(realpath-resolved, prefix allowlist, fail-closed) for ``file://`` / bare
paths, raising ``MediaRefError`` — the contract ``rag_system``'s own media
fetch already raised.  The tests here pin that end to end, offline:

  * ``file:///etc/passwd`` and bare ``/etc/passwd`` refused (async fetch,
    sync twin, video-frames path, standalone guard, policy layer);
  * a path inside the default prefixes (``{DATA_PATH}/datasets/...``,
    ``{DATA_PATH}/staging/...``) still fetches — the happy path is intact;
  * a path under ``{DATA_PATH}`` but NOT under the prefixes (the identity
    store at ``{DATA_PATH}/access/alice.json``!) is refused;
  * a ``..``-traversal that realpaths outside the prefixes is refused;
  * ``data:`` refs stay inert (never read from disk);
  * the server's own ``MEDIA_BASE_URL`` host still passes the http(s)
    policy check with NO network touched (getaddrinfo monkeypatched to
    fail loudly if called);
  * ``_classify_url`` no longer sniffs (opens) out-of-prefix files.

media_paths freezes its prefix tuple from ``DATA_PATH`` at import time, so —
the house pattern from ``tests/security/test_security_kernel.py`` — the
tests monkeypatch the module attributes (with ``DATA_PATH`` set as well, in
the ``test_access_store.py`` style, so error messages render the tmp tree).

Run::

    pytest tests/security/test_media_fetch_allowlist.py -q
"""

import asyncio
import base64
import os
import socket
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.utils.media_paths as mp
from multimodal_rag.utils import url_policy as up
from multimodal_rag.utils.media_paths import MediaRefError
from multimodal_rag.utils.model_adapters import InputConversion, _classify_url, _guard_media_fetch

_JPEG = b"\xff\xd8\xff\xe0" + b"JFIF-payload" * 8


@pytest.fixture
def media_env(tmp_path, monkeypatch):
    """Allowlist pointed at tmp: {DATA_PATH}/datasets + {DATA_PATH}/staging.

    media_paths reads DATA_PATH and MEDIA_ALLOW_PATH_PREFIXES once at
    import, so setenv alone cannot re-aim it — patch the module attributes
    (test_security_kernel.py's documented approach) and set DATA_PATH too so
    the refusal messages name the tmp tree.
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


def _conv() -> InputConversion:
    """An InputConversion over a stub embedder — no endpoint, no resize."""
    emb = SimpleNamespace(mm_processor_kwargs={"max_pixels": 0}, http_async_client=None)
    return InputConversion(emb)


def _write_real_jpeg(path) -> None:
    """A real (1x1) JPEG — the sync-twin image branch re-encodes via PIL, so
    a fake header would be skipped as 'unreadable' instead of embedded."""
    from PIL import Image

    Image.new("RGB", (1, 1), (200, 30, 30)).save(path, format="JPEG")


def _dataset_image(root, name="a.jpg"):
    img = root / "datasets" / "ds" / "files" / name
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(_JPEG)
    return img


# ---------------------------------------------------------------------------
# (a) file:///etc/passwd — the audit's exact repro — refused everywhere
# ---------------------------------------------------------------------------


def test_async_fetch_refuses_etc_passwd(media_env):
    conv = _conv()
    with pytest.raises(MediaRefError, match="outside the allowed prefixes"):
        asyncio.run(conv._fetch_obj_async("file:///etc/passwd"))
    # Bare-path form of the same read is equally refused.
    with pytest.raises(MediaRefError):
        asyncio.run(conv._fetch_obj_async("/etc/passwd"))


def test_policy_layer_refuses_etc_passwd(media_env):
    # url_policy used to return silently for non-http(s) — the P1-3 half of
    # the hole (every entry gate funneling through it waved the ref through).
    with pytest.raises(MediaRefError, match="outside the allowed prefixes"):
        up._check_media_url_policy("file:///etc/passwd")
    with pytest.raises(MediaRefError):
        up._check_media_url_policy("/etc/passwd")


def test_standalone_guard_refuses_etc_passwd(media_env):
    with pytest.raises(MediaRefError):
        _guard_media_fetch("file:///etc/passwd")
    with pytest.raises(MediaRefError):
        _guard_media_fetch("/etc/passwd")


def test_video_frames_path_refuses_etc_passwd(media_env):
    conv = _conv()
    with pytest.raises(MediaRefError):
        asyncio.run(conv._fetch_video_frames("file:///etc/passwd"))


# ---------------------------------------------------------------------------
# (b) inside the default prefixes → ALLOWED (happy path intact, fully offline)
# ---------------------------------------------------------------------------


def test_async_fetch_allows_dataset_path(media_env):
    img = _dataset_image(media_env)
    conv = _conv()
    b64, mime = asyncio.run(conv._fetch_obj_async(str(img)))
    assert mime == "image/jpeg"
    assert base64.b64decode(b64) == _JPEG
    # The file:// form of the same allowed path is equivalent.
    b64b, _ = asyncio.run(conv._fetch_obj_async(f"file://{img}"))
    assert base64.b64decode(b64b) == _JPEG


def test_async_fetch_allows_staging_path(media_env):
    snd = media_env / "staging" / "upload.jpg"
    snd.write_bytes(_JPEG)
    b64, mime = asyncio.run(_conv()._fetch_obj_async(str(snd)))
    assert mime == "image/jpeg" and base64.b64decode(b64) == _JPEG


def test_sync_twin_allows_dataset_path(media_env):
    img = media_env / "datasets" / "ds" / "files" / "b.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    _write_real_jpeg(img)
    out = _conv()._add_raw_url(
        [{"role": "user", "content": [], "_audio_urls": [], "_image_urls": [str(img)], "_video_urls": []}]
    )
    url = out[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# (c) under DATA_PATH but NOT under the prefixes → REFUSED (the identity
#     store at {DATA_PATH}/access/ is the sensitive neighbor)
# ---------------------------------------------------------------------------


def test_data_path_outside_prefixes_refused(media_env):
    access = media_env / "access" / "alice.json"
    access.parent.mkdir(parents=True, exist_ok=True)
    access.write_text('{"k":"v"}')
    conv = _conv()
    for ref in (str(access), f"file://{access}"):
        with pytest.raises(MediaRefError, match="outside the allowed prefixes"):
            asyncio.run(conv._fetch_obj_async(ref))
        with pytest.raises(MediaRefError):
            up._check_media_url_policy(ref)
        with pytest.raises(MediaRefError):
            _guard_media_fetch(ref)
    # The file EXISTS — only the allowlist stands between it and the read.
    assert access.exists()


# ---------------------------------------------------------------------------
# (d) traversal — realpath resolves it outside the prefixes → REFUSED
# ---------------------------------------------------------------------------


def test_traversal_refused_by_realpath(media_env):
    # {media_env}/datasets/ds/../../access/clients.json realpaths to
    # {media_env}/access/clients.json — outside every allowed prefix
    # (realpath needs no existence, so the target need not be created).
    evil = f"file://{media_env}/datasets/ds/../../access/clients.json"
    conv = _conv()
    with pytest.raises(MediaRefError, match="outside the allowed prefixes"):
        asyncio.run(conv._fetch_obj_async(evil))
    with pytest.raises(MediaRefError):
        up._check_media_url_policy(evil)


# ---------------------------------------------------------------------------
# (e) the sync twin refuses exactly like the async fetch — on ALL media keys
# ---------------------------------------------------------------------------


def test_sync_twin_refuses_like_async(media_env):
    conv = _conv()
    for key in ("_audio_urls", "_image_urls", "_video_urls"):
        req = [{"role": "user", "content": [], "_audio_urls": [], "_image_urls": [], "_video_urls": []}]
        req[0][key] = ["file:///etc/passwd"]
        with pytest.raises(MediaRefError):
            conv._add_raw_url(req)
    # The image refusal must PROPAGATE, not be swallowed by the branch's
    # skip-unreadable-image handler (the guard sits before that try block).
    req = [{"role": "user", "content": [], "_audio_urls": [], "_image_urls": ["/etc/passwd"], "_video_urls": []}]
    with pytest.raises(MediaRefError):
        conv._add_raw_url(req)


# ---------------------------------------------------------------------------
# (f) MEDIA_BASE_URL self-reference still passes — with NO network touched
# ---------------------------------------------------------------------------


def test_own_media_base_url_policy_check_needs_no_network(monkeypatch):
    monkeypatch.setenv("MEDIA_BASE_URL", "http://media.internal:8000")

    def _no_dns(*a, **k):
        raise AssertionError("the media policy check must not touch DNS/network")

    monkeypatch.setattr(socket, "getaddrinfo", _no_dns)
    url = "http://media.internal:8000/api/datasets/d/files/a.jpg?token=t"
    up._check_media_url_policy(url)  # must not raise, must not resolve
    _guard_media_fetch(url)  # the embedder-layer guard: same verdict
    # A DIFFERENT private host stays refused (literal IP — classified
    # without DNS, so the no-network patch above still holds).
    with pytest.raises(ValueError):
        _guard_media_fetch("http://10.9.9.9/x.jpg")


# ---------------------------------------------------------------------------
# Inert classes + documented escapes, pinned so they survive refactors
# ---------------------------------------------------------------------------


def test_data_urls_stay_inert(media_env):
    # data: refs are never read from disk — guard, policy and classify all
    # pass/see them without touching the filesystem.
    ref = "data:image/png;base64,AA=="
    _guard_media_fetch(ref)
    up._check_media_url_policy(ref)
    assert _classify_url(ref) == "image"


def test_fail_closed_when_no_prefixes_configured(tmp_path, monkeypatch):
    """Explicitly empty MEDIA_ALLOW_PATH_PREFIXES allows nothing."""
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_PATH_PREFIXES", ())
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_ANY", False)
    with pytest.raises(MediaRefError):
        _guard_media_fetch(str(tmp_path / "datasets" / "x.jpg"))


def test_star_escape_hatch_allows_anything(monkeypatch):
    """The documented dev/test escape (prefix ``*``) still passes the guard —
    pinning that the escape lives in media_paths, not re-implemented here."""
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_PATH_PREFIXES", ("*",))
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_ANY", True)
    _guard_media_fetch("file:///etc/passwd")
    _guard_media_fetch("/etc/passwd")


# ---------------------------------------------------------------------------
# _classify_url must not sniff out-of-prefix files (the pre-guard 32-byte read)
# ---------------------------------------------------------------------------


def test_classify_url_does_not_sniff_outside_prefixes(media_env):
    secret = media_env / "access" / "probe.jpg"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_bytes(_JPEG)  # real JPEG magic — the old code happily sniffed it
    # Out-of-prefix existing file: NOT classified as media (no read), so a
    # bare-path string input is treated as plain text instead of being
    # routed to the fetcher.
    assert _classify_url(str(secret)) is None
    # The same file inside the prefixes classifies normally.
    allowed = _dataset_image(media_env, "c.jpg")
    assert _classify_url(str(allowed)) == "image"
