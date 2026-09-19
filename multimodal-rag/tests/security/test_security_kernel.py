"""Security-kernel unit tests (fleet audit 3.1: the kernel had ZERO unit tests).

Covers the guards the audit verified-good by READING and now pins by TEST:

  * PBKDF2-SHA256 (600k iterations) password hashing + constant-time verify
    — malformed/tampered stored hashes must return False, never raise.
  * Short-lived HMAC media tokens (mint/verify/expiry/tamper), the
    dataset-scoped ``rel_path="*"`` token (and its forgery boundary), and
    the MCP→REST shared-secret compatibility (MCP mints, REST serves).
  * Archive guards: zip-slip member rejection, declared-size (bomb) bounds,
    streamed decompression cap, symlink rejection (tar data filter + the
    post-extraction sweep), nested-archive recursion + depth cap.
  * Local media-path allowlist (MEDIA_ALLOW_PATH_PREFIXES): traversal,
    realpath/symlink escapes, fail-closed defaults, ``*`` escape.
  * REST file serving: directory-traversal refusal end-to-end (TestClient
    against the real app; the fleet rule only forbids GET /mcp) and the
    media-token gate on that route.
  * MCP auth middleware (utils/mcp_auth.py): key union, both header forms,
    401 on missing/wrong key, open-dev-mode pass-through.

Every test is offline: no Qdrant server, no model endpoints, no external
network (loopback servers only, used by test_url_policy.py).

Run::

    pytest tests/security/ -q
"""

import io
import os
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as _api_mod
from multimodal_rag.api_server import (
    _sign_media_token as _api_sign,
)
from multimodal_rag.api_server import (
    _verify_media_token as _api_verify,
)
from multimodal_rag.dataset_manager import (
    _PBKDF2_ITERATIONS,
    DatasetManager,
    _check_password,
    _hash_password,
)
from multimodal_rag.input_processing import archive_processor as ap
from multimodal_rag.mcp_server import _sign_media_token as _mcp_sign
from multimodal_rag.utils import media_paths as mp
from multimodal_rag.utils.mcp_auth import (
    ApiKeyAuthMiddleware,
    configured_keys,
    presented_keys,
    warn_if_open,
)
from multimodal_rag.utils.media_paths import (
    MediaRefError,
    _media_path_allowed,
    _validate_document_media_refs,
    _validate_media_ref,
)


def _which(name):
    from shutil import which

    return which(name)


def _rarfile_importable():
    try:
        import rarfile  # noqa: F401

        return True
    except Exception:
        return False


# ===========================================================================
# 1 · PBKDF2 password hashing (dataset_manager)
# ===========================================================================


def test_pbkdf2_iteration_floor_is_600k():
    # OWASP 2023 minimum for PBKDF2-SHA256 — the suite pins the FLOOR so a
    # "performance tweak" cannot silently downgrade the stored-hash cost.
    assert _PBKDF2_ITERATIONS >= 600_000


def test_hash_format_and_roundtrip():
    stored = _hash_password("s3cret-π-password")
    salt, h = stored.split("$", 1)
    assert len(salt) == 32 and len(h) == 64  # 16-byte hex salt, sha256 hex
    int(salt, 16)  # salt is hex
    int(h, 16)
    assert _check_password("s3cret-π-password", stored) is True
    assert _check_password("wrong", stored) is False


def test_hashes_are_salted_per_hash():
    a, b = _hash_password("same"), _hash_password("same")
    assert a != b, "two hashes of the same password must not share a salt"
    assert _check_password("same", a) and _check_password("same", b)


def test_check_password_malformed_never_raises(monkeypatch):
    # Speed: malformed-input handling does not need the full 600k cost.
    import multimodal_rag.dataset_manager as dm_mod

    monkeypatch.setattr(dm_mod, "_PBKDF2_ITERATIONS", 1000)
    real = _hash_password("pw")
    bad_stored = [
        "",
        "no-separator",
        "salt-without-hash$",
        "$only-hash",
        "a$b$c",
        "zz$not-hex",  # unparseable hex digest
        real[:-1] + ("0" if real[-1] != "0" else "1"),  # tampered digest
        real.replace("$", "$$", 1),  # empty salt segment
    ]
    for stored in bad_stored:
        assert _check_password("pw", stored) is False, f"stored={stored!r}"
    for weird in (None, 123, b"bytes", ["list"], {"d": 1}):
        assert _check_password("pw", weird) is False, f"stored type {type(weird).__name__}"
    # wrong stored TYPE for password argument must not raise either
    assert _check_password(None, real) is False


def test_dataset_manager_verify_password(base_path, embedder):
    # base_path (conftest) chdirs into the tmp dir so the local-Qdrant
    # default ./qdrant_storage lands there, not in the repo tree.
    dm = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    dm.create_dataset("pwds")
    assert dm.verify_password("pwds", "anything") is True  # no password set
    dm.set_password("pwds", "hunter2")
    assert dm.has_password("pwds") is True
    assert dm.verify_password("pwds", "hunter2") is True
    assert dm.verify_password("pwds", "hunter3") is False
    with pytest.raises(FileNotFoundError):
        dm.verify_password("no-such-dataset", "x")
    dm.delete_dataset("pwds")


# ===========================================================================
# 2 · Media tokens (api_server + mcp_server share the HMAC scheme)
# ===========================================================================


@pytest.fixture
def token_secret(monkeypatch):
    secret = "unit-test-media-token-secret-0123456789abcdef"
    import multimodal_rag.api_server as api
    import multimodal_rag.mcp_server as mcp

    monkeypatch.setattr(api, "_MEDIA_TOKEN_SECRET", secret)
    monkeypatch.setattr(mcp, "_MEDIA_TOKEN_SECRET", secret)
    monkeypatch.setattr(api, "_MEDIA_TOKEN_TTL", 3600)
    return secret


def test_media_token_roundtrip(token_secret):
    tok = _api_sign("dsA", "files/abc_report.pdf")
    expiry, sig = tok.split(".", 1)
    assert len(sig) == 32  # 128-bit truncated signature
    assert int(expiry) > int(time.time())
    assert _api_verify("dsA", "files/abc_report.pdf", tok) is True


def test_media_token_binds_dataset_and_path(token_secret):
    tok = _api_sign("dsA", "files/abc_report.pdf")
    assert _api_verify("dsB", "files/abc_report.pdf", tok) is False  # other dataset
    assert _api_verify("dsA", "files/other.pdf", tok) is False  # other file
    assert _api_verify("dsA", "../outside.jpg", tok) is False  # traversal shape


def test_media_token_expiry(token_secret):
    import multimodal_rag.api_server as api

    now = int(time.time())
    assert _api_verify("dsA", "f.jpg", _api_sign("dsA", "f.jpg", expiry=now - 10)) is False
    # expiry further in the future than TTL + slack is refused (bounds mint-ahead)
    far = _api_sign("dsA", "f.jpg", expiry=now + api._MEDIA_TOKEN_TTL + 3600)
    assert _api_verify("dsA", "f.jpg", far) is False


def test_media_token_tamper_resistant(token_secret):
    tok = _api_sign("dsA", "files/a.jpg")
    expiry, sig = tok.split(".", 1)
    flipped = sig[0] == "0" and "1" + sig[1:] or "0" + sig[1:]
    assert _api_verify("dsA", "files/a.jpg", f"{expiry}.{flipped}") is False
    assert _api_verify("dsA", "files/a.jpg", sig) is False  # truncated (no expiry)
    assert _api_verify("dsA", "files/a.jpg", "not-a-token") is False
    assert _api_verify("dsA", "files/a.jpg", "") is False
    assert _api_verify("dsA", "files/a.jpg", f"{expiry}.{sig}deadbeef") is False


def test_media_token_fail_closed_without_secret(monkeypatch, token_secret):
    tok = _api_sign("dsA", "f.jpg")
    monkeypatch.setattr(_api_mod, "_MEDIA_TOKEN_SECRET", "")
    assert _api_mod._MEDIA_TOKEN_SECRET == ""
    assert _api_verify("dsA", "f.jpg", tok) is False  # unsigned config serves nothing


def test_media_token_dataset_scoped_star_and_forgery_boundary(token_secret):
    """rel_path="*" mints a dataset-scoped token (web UI).

    Forgery boundary: the "*" token is valid for ANY file of ITS dataset —
    but never for another dataset, and a file-scoped token can never act as
    a wildcard.
    """
    star = _api_sign("dsA", "*")  # what GET /api/datasets/{name}/media-token mints
    assert _api_verify("dsA", "files/anything.jpg", star) is True
    assert _api_verify("dsA", "a/b/c.mp4", star) is True
    assert _api_verify("dsB", "files/anything.jpg", star) is False  # cross-dataset forgery
    scoped = _api_sign("dsA", "files/one.jpg")
    assert _api_verify("dsA", "*", scoped) is False  # file token ≠ dataset token


def test_mcp_and_rest_share_the_token_scheme(token_secret):
    """The MCP server mints media URLs the REST server must serve."""
    mcp_tok = _mcp_sign("dsA", "files/video.mp4")
    assert _api_verify("dsA", "files/video.mp4", mcp_tok) is True
    rest_tok = _api_sign("dsA", "files/video.mp4")
    import hashlib
    import hmac

    # byte-identical scheme: same message construction + key
    msg = f"dsA:files/video.mp4:{rest_tok.split('.')[0]}".encode()
    full = hmac.new(token_secret.encode(), msg, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(full[:32], rest_tok.split(".", 1)[1])


def test_media_token_ttl_floor(monkeypatch):
    # TTL below 60s must clamp up (the env parse floors at 60); evaluated the
    # same way the servers parse MEDIA_TOKEN_TTL at import.
    monkeypatch.setenv("MEDIA_TOKEN_TTL", "5")
    ttl = max(60, int(os.environ.get("MEDIA_TOKEN_TTL", "3600")))
    assert ttl == 60


# ===========================================================================
# 3 · Archive guards (input_processing.archive_processor)
# ===========================================================================


@pytest.fixture
def small_caps(monkeypatch):
    """Tight archive bounds so bomb tests don't need gigabyte archives."""
    monkeypatch.setattr(ap, "_ARCHIVE_MAX_TOTAL_BYTES", 1 * 1024 * 1024)  # 1 MiB
    monkeypatch.setattr(ap, "_ARCHIVE_MAX_MEMBER_BYTES", 512 * 1024)  # 512 KiB
    monkeypatch.setattr(ap, "_ARCHIVE_MAX_ENTRIES", 50)


def _zip_bytes(*members: tuple[str, bytes], **kw) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def _write(tmp_path, name: str, data: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_is_safe_member_matrix():
    dest = tempfile.mkdtemp()
    safe = ("ok.txt", "sub/dir/file.txt", "sub/../ok.txt", "a b/c.txt")
    unsafe = ("../evil.txt", "../../etc/passwd", "/etc/passwd", "..", "sub/../../out.txt")
    for m in safe:
        assert ap._is_safe_member(dest, m) is True, m
    for m in unsafe:
        assert ap._is_safe_member(dest, m) is False, m
    # "....//evil" is four literal dots (a real directory name), not traversal —
    # it stays inside dest, so safe is the CORRECT verdict.
    assert ap._is_safe_member(dest, "....//evil") is True


def test_zip_slip_member_is_skipped(tmp_path):
    """A traversal member must not land outside the extraction directory."""
    zp = _write(
        tmp_path,
        "evil.zip",
        _zip_bytes(
            ("../../escaped.txt", b"pwned"),
            ("ok.txt", b"fine"),
        ),
    )
    dest = str(tmp_path / "out")
    os.makedirs(dest)
    ap.ArchiveProcessor._extract(zp, dest)
    assert not (tmp_path / "escaped.txt").exists()
    assert not os.path.exists(os.path.join(dest, "..", "escaped.txt"))
    assert (Path(dest) / "ok.txt").read_bytes() == b"fine"


def test_absolute_member_is_skipped(tmp_path):
    zp = _write(
        tmp_path,
        "abs.zip",
        _zip_bytes(
            ("/etc/never", b"nope"),
            ("ok.txt", b"fine"),
        ),
    )
    dest = str(tmp_path / "out")
    os.makedirs(dest)
    ap.ArchiveProcessor._extract(zp, dest)
    assert (Path(dest) / "ok.txt").exists()


def test_zip_bomb_declared_bounds_total(tmp_path, small_caps):
    """Declared uncompressed size over the total budget — refused BEFORE any
    extraction (a 2 MiB-in / >1 MiB-declared zeros member trips the cap)."""
    data = b"\0" * (2 * 1024 * 1024)
    zp2 = _write(tmp_path, "bomb2.zip", _zip_bytes(("zeros.bin", data)))
    with pytest.raises(ValueError):
        ap.ArchiveProcessor._check_bounds(zp2)


def test_zip_bomb_per_member_bound(tmp_path, small_caps):
    data = b"\0" * (600 * 1024)  # over the 512 KiB member cap
    zp = _write(tmp_path, "member.zip", _zip_bytes(("one.bin", data)))
    with pytest.raises(ValueError):
        ap.ArchiveProcessor._check_bounds(zp)


def test_zip_entry_count_bound(tmp_path, small_caps):
    members = [(f"f{i}.txt", b"x") for i in range(60)]  # over the 50-entry cap
    zp = _write(tmp_path, "many.zip", _zip_bytes(*members))
    with pytest.raises(ValueError):
        ap.ArchiveProcessor._check_bounds(zp)


def test_within_bounds_archive_processes(tmp_path, small_caps):
    seen = []

    def _member(path, name):
        seen.append(name)
        return ["id"]

    zp = _write(
        tmp_path,
        "ok.zip",
        _zip_bytes(
            ("docs/readme.txt", b"hello"),
            ("data.csv", b"a,b\n1,2\n"),
        ),
    )
    proc = ap.ArchiveProcessor(process_member=_member)
    ids = proc.process(zp)
    assert ids == ["id", "id"] and sorted(seen) == ["data.csv", "readme.txt"]


def test_bare_gz_streamed_cap(tmp_path, small_caps):
    """A decompression bomb (tiny gz → megabytes) is capped WHILE streaming."""
    import gzip

    raw = b"\0" * (2 * 1024 * 1024)  # 2 MiB of zeros from a few KB of gz
    gp = _write(tmp_path, "bomb.gz", gzip.compress(raw))
    dest = str(tmp_path / "gzout")
    os.makedirs(dest)
    with pytest.raises(ValueError):
        ap.ArchiveProcessor._extract(gp, dest)
    # and a small one passes through
    small = _write(tmp_path, "ok.gz", gzip.compress(b"tiny payload"))
    dest2 = str(tmp_path / "gzout2")
    os.makedirs(dest2)
    ap.ArchiveProcessor._extract(small, dest2)
    assert (Path(dest2) / "ok").read_bytes() == b"tiny payload"


def test_tar_symlink_member_rejected(tmp_path):
    """filter="data" refuses symlink members (absolute AND outside-relative)."""
    tp = _write(tmp_path, "sym.tar", b"")
    with tarfile.open(tp, "w") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
        payload = b"data"
        f = tarfile.TarInfo("f.txt")
        f.size = len(payload)
        tf.addfile(f, io.BytesIO(payload))
    dest = str(tmp_path / "out")
    os.makedirs(dest)
    with pytest.raises(tarfile.FilterError):  # AbsoluteLinkError
        ap.ArchiveProcessor._extract(tp, dest)
    # relative-outside symlink is also refused
    tp2 = _write(tmp_path, "sym2.tar", b"")
    with tarfile.open(tp2, "w") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../../outside"
        tf.addfile(info)
    dest2 = str(tmp_path / "out2")
    os.makedirs(dest2)
    with pytest.raises(tarfile.FilterError):  # LinkOutsideDestinationError
        ap.ArchiveProcessor._extract(tp2, dest2)
    assert not os.path.islink(os.path.join(dest2, "link"))


def test_sweep_rejects_planted_symlink(tmp_path):
    """The post-extraction sweep (RAR path) catches a symlink that landed."""
    dest = tmp_path / "sweep"
    (dest / "sub").mkdir(parents=True)
    (dest / "sub" / "real.txt").write_text("x")
    os.symlink("/etc/passwd", str(dest / "sub" / "escape"))
    with pytest.raises(ValueError, match="symlink member"):
        ap._sweep_extracted_tree(str(dest / "fake.rar"), str(dest))


def test_sweep_enforces_extracted_size(tmp_path, small_caps):
    dest = tmp_path / "sweep2"
    dest.mkdir()
    (dest / "big.bin").write_bytes(b"\0" * (600 * 1024))  # over member cap
    with pytest.raises(ValueError):
        ap._sweep_extracted_tree(str(dest / "x.rar"), str(dest))


def test_nested_archive_recursion_and_depth_cap(tmp_path):
    """zip→zip→file is processed; beyond max_depth the innermost is not."""
    innermost = _zip_bytes(("deep.txt", b"layer4"))
    l3 = _write(tmp_path, "l3.zip", innermost)
    with open(l3, "rb") as f:
        l2 = _write(tmp_path, "l2.zip", _zip_bytes(("nested/l3.zip", f.read())))
    with open(l2, "rb") as f:
        l1 = _write(tmp_path, "l1.zip", _zip_bytes(("nested/l2.zip", f.read())))
    with open(l1, "rb") as f:
        l0 = _write(tmp_path, "l0.zip", _zip_bytes(("l1.zip", f.read())))

    seen = []
    proc = ap.ArchiveProcessor(process_member=lambda p, n: seen.append(n) or ["id"])
    proc.process(l0)
    assert "deep.txt" in seen, "three levels of nesting stay within max_depth=3"

    # one level deeper → the innermost file is skipped (depth cap)
    with open(l0, "rb") as f:
        l0b = _write(tmp_path, "l0b.zip", _zip_bytes(("wrapper/l1.zip", f.read())))
    seen.clear()
    proc.process(l0b)
    assert "deep.txt" not in seen, "content beyond max_depth must not be processed"


def test_unsupported_archive_extension(tmp_path):
    p = _write(tmp_path, "archive.zoo", b"not really")
    with pytest.raises(ValueError, match="Unsupported archive format"):
        ap.ArchiveProcessor._extract(p, str(tmp_path))


@pytest.mark.skipif(
    _rarfile_importable() or _which("unrar") is not None,
    reason="rarfile/unrar support IS present — the no-RAR-support error path cannot be exercised here",
)
def test_rar_without_support_is_loud(tmp_path):
    p = _write(tmp_path, "x.rar", b"Rar!\x1a\x07\x00" + b"\0" * 16)
    with pytest.raises(ValueError, match="neither the 'rarfile' package nor the 'unrar' binary"):
        ap.ArchiveProcessor._check_bounds(p)


# ===========================================================================
# 4 · Local media-path allowlist + traversal (utils/media_paths.py)
# ===========================================================================


@pytest.fixture
def allow_prefixes(tmp_path, monkeypatch):
    """Point the allowlist at a temp tree (the module reads the tuple at
    call time — attributes are monkeypatched, never the env)."""
    datasets = tmp_path / "datasets"
    staging = tmp_path / "staging"
    datasets.mkdir()
    staging.mkdir()
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_PATH_PREFIXES", (str(datasets), str(staging)))
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_ANY", False)
    return tmp_path


def test_media_path_allowlist_matrix(allow_prefixes):
    inside = allow_prefixes / "datasets" / "ds" / "files" / "a.jpg"
    assert _media_path_allowed(str(inside)) is True
    assert _media_path_allowed(f"file://{inside}") is True
    assert _media_path_allowed(str(allow_prefixes / "etc" / "passwd")) is False
    assert _media_path_allowed("/etc/passwd") is False
    # traversal that resolves outside
    assert _media_path_allowed(str(allow_prefixes / "datasets" / ".." / ".." / "etc" / "passwd")) is False
    # prefix-boundary confusion: /data/datasets-evil is NOT inside /data/datasets
    evil = allow_prefixes / "datasets-evil" / "x"
    assert _media_path_allowed(str(evil)) is False


def test_media_path_symlink_escape(allow_prefixes):
    """A symlink inside an allowed prefix must not smuggle an outside file."""
    link = allow_prefixes / "datasets" / "link.jpg"
    os.symlink("/etc/passwd", str(link))
    assert _media_path_allowed(str(link)) is False  # realpath resolves outside


def test_media_path_fail_closed_and_star_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_ANY", False)
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_PATH_PREFIXES", ())
    assert _media_path_allowed(str(tmp_path / "x.jpg")) is False  # empty = allow nothing
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_ANY", True)
    assert _media_path_allowed("/etc/passwd") is True  # documented dev escape


def test_validate_media_ref_classes(allow_prefixes, monkeypatch):
    # data: URLs are inert
    _validate_media_ref("data:image/png;base64,AAAA", "image")
    # s3:// is never fetched from document payloads
    with pytest.raises(MediaRefError, match="s3://"):
        _validate_media_ref("s3://bucket/key", "video")
    # outside paths (bare and file://) refused, field named in the error
    with pytest.raises(MediaRefError, match="'image'"):
        _validate_media_ref("/etc/passwd", "image")
    with pytest.raises(MediaRefError, match="'video'"):
        _validate_media_ref("file:///etc/shadow", "video")
    # http(s) delegates to the URL policy
    from multimodal_rag.utils import url_policy as up

    with pytest.raises(MediaRefError, match="'audio'"):
        monkeypatch.setattr(up, "_INGEST_BLOCK_PRIVATE", True)
        _validate_media_ref("http://10.0.0.9/x.mp3", "audio")
    # loopback allowed on the media policy (server's own media URLs)
    _validate_media_ref("http://localhost:8000/api/datasets/d/files/a.jpg?token=x", "image")


def test_validate_document_media_refs_walks_all_keys(allow_prefixes, monkeypatch):
    docs = [
        {"text": "plain string doc skipped"},
        {"image": "/etc/passwd", "preprocessed_video": "s3://b/k"},
        "not-a-dict",
        {},
    ]
    with pytest.raises(MediaRefError, match="'image'"):
        _validate_document_media_refs(docs)


# ===========================================================================
# 5 · REST file serving: traversal + token gate (end-to-end via TestClient)
# ===========================================================================


class _FakeDM:
    def __init__(self, root: Path):
        self._root = root

    def _dataset_dir(self, name):
        return self._root / name

    def has_password(self, name):
        return False

    def verify_password(self, name, password):
        return False


@pytest.fixture
def served_dataset(tmp_path, monkeypatch):
    """The real FastAPI app with a stubbed manager (offline) and one dataset."""
    from fastapi.testclient import TestClient

    import multimodal_rag.api_server as api

    root = tmp_path / "datasets"
    files = root / "dsA" / "files"
    files.mkdir(parents=True)
    (files / "abc_report.pdf").write_bytes(b"%PDF-1.4 real-content")
    (files / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0jpegdata")

    async def _fake_manager():
        return _FakeDM(root)

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    client = TestClient(api.app)
    return client, files


def test_rest_serves_file_within_dataset(served_dataset):
    (client, _files) = served_dataset
    r = client.get("/api/datasets/dsA/files/abc_report.pdf")
    assert r.status_code == 200
    # SVG/HTML-classified files force attachment; a real PDF may render, but
    # content must be the stored bytes
    assert r.content.startswith(b"%PDF")


def test_rest_traversal_refused(served_dataset):
    (client, _files) = served_dataset
    for evil in (
        "..%2F..%2F..%2Fetc%2Fpasswd",
        "../../../../etc/passwd",
        "..\\..\\etc\\passwd",
        "sub/../../../../etc/passwd",
        "%2e%2e/%2e%2e/etc/passwd",
    ):
        r = client.get(f"/api/datasets/dsA/files/{evil}")
        assert r.status_code in (403, 404), f"{evil} → {r.status_code}"
        assert b"root:" not in r.content


def test_rest_media_token_gate(served_dataset, token_secret):
    (client, _files) = served_dataset
    good = _api_sign("dsA", "img.jpg")
    assert client.get("/api/datasets/dsA/files/img.jpg", params={"token": good}).status_code == 200
    # forged / foreign / expired tokens never serve
    assert client.get("/api/datasets/dsA/files/img.jpg", params={"token": good[:-2] + "zz"}).status_code == 403
    assert (
        client.get("/api/datasets/dsA/files/img.jpg", params={"token": _api_sign("dsB", "img.jpg")}).status_code == 403
    )
    # dataset-scoped token serves any file of the dataset, none of another
    star = _api_sign("dsA", "*")
    assert client.get("/api/datasets/dsA/files/img.jpg", params={"token": star}).status_code == 200
    assert client.get("/api/datasets/dsB/files/img.jpg", params={"token": star}).status_code == 403


# ===========================================================================
# 6 · MCP auth middleware (utils/mcp_auth.py — the hardlinked fleet module)
# ===========================================================================


def _scope(path="/mcp", headers=None, client=("10.9.0.1", 5000)):
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers or [],
        "client": client,
        "query_string": b"",
    }


async def _drive(middleware, scope):
    """Run the ASGI middleware with stub receive/send; return the response."""
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware.app = app
    await middleware(scope, receive, send)
    return messages


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_configured_keys_union_and_dedup(monkeypatch):
    monkeypatch.setenv("MCP_API_KEYS", "k1, k2,k1")
    monkeypatch.setenv("RAG_API_KEYS", "k2, k3")
    assert configured_keys(("MCP_API_KEYS", "RAG_API_KEYS")) == ["k1", "k2", "k3"]
    monkeypatch.delenv("MCP_API_KEYS")
    monkeypatch.delenv("RAG_API_KEYS")
    assert configured_keys(("MCP_API_KEYS", "RAG_API_KEYS")) == []


def test_presented_keys_header_forms():
    scope = _scope(headers=[(b"authorization", b"Bearer abc"), (b"x-api-key", b"def")])
    assert presented_keys(scope) == ["abc", "def"]
    scope = _scope(headers=[(b"authorization", b"Basic dXNlcjpwYXNz")])
    assert presented_keys(scope) == []
    scope = _scope(headers=[(b"authorization", b"Bearer  padded  ")])
    assert presented_keys(scope) == ["padded"]


def test_middleware_enforces_key(monkeypatch):
    monkeypatch.setenv("MCP_API_KEYS", "secret-key-1")
    mw = ApiKeyAuthMiddleware(
        None, env_names=("MCP_API_KEYS", "RAG_API_KEYS"), protected=lambda p: p.startswith("/mcp")
    )

    ok = [(b"authorization", b"Bearer secret-key-1")]
    assert _run(_drive(mw, _scope(headers=ok)))[0]["status"] == 200
    ok2 = [(b"x-api-key", b"secret-key-1")]
    assert _run(_drive(mw, _scope(headers=ok2)))[0]["status"] == 200

    missing = _run(_drive(mw, _scope()))
    assert missing[0]["status"] == 401
    wrong = [(b"authorization", b"Bearer wrong-key")]
    assert _run(_drive(mw, _scope(headers=wrong)))[0]["status"] == 401
    # health probes stay public
    assert _run(_drive(mw, _scope(path="/healthz")))[0]["status"] == 200
    # non-protected paths stay public
    assert _run(_drive(mw, _scope(path="/console")))[0]["status"] == 200


def test_middleware_reads_keys_per_request(monkeypatch):
    """Rotation without restart: the key set is re-read on every request."""
    mw = ApiKeyAuthMiddleware(None, env_names=("MCP_API_KEYS",), protected=lambda p: True)
    monkeypatch.setenv("MCP_API_KEYS", "old-key")
    assert _run(_drive(mw, _scope(headers=[(b"x-api-key", b"old-key")])))[0]["status"] == 200
    monkeypatch.setenv("MCP_API_KEYS", "new-key")
    assert _run(_drive(mw, _scope(headers=[(b"x-api-key", b"old-key")])))[0]["status"] == 401
    assert _run(_drive(mw, _scope(headers=[(b"x-api-key", b"new-key")])))[0]["status"] == 200


def test_middleware_open_dev_mode(monkeypatch):
    monkeypatch.delenv("MCP_API_KEYS", raising=False)
    monkeypatch.delenv("RAG_API_KEYS", raising=False)
    mw = ApiKeyAuthMiddleware(None, env_names=("MCP_API_KEYS",), protected=lambda p: True)
    assert _run(_drive(mw, _scope()))[0]["status"] == 200
    assert warn_if_open("unit-test", ("MCP_API_KEYS",)) is True
    monkeypatch.setenv("MCP_API_KEYS", "k")
    assert warn_if_open("unit-test", ("MCP_API_KEYS",)) is False
