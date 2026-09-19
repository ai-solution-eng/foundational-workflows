"""url_policy tests: deny classes + the DNS-rebinding pin (fleet audit MED).

The audit's MED finding on ``utils/url_policy.py:56-91``: the policy checked
DNS at CHECK time while the fetch resolved DNS again at FETCH time — a
rebinding answer in between reroutes the request into private space.  The
fix (this wave): :func:`validate_fetch_url` resolves ONCE, classifies that
exact address set, and hands back a :class:`PinnedUrl` whose connection
targets a member of the validated set (Host header + TLS SNI preserved).

Tests cover:

  * the deny classes on literal IPs and on monkeypatched resolutions —
    loopback / RFC1918 / link-local+metadata / unspecified / multicast /
    reserved / ULA, plus ``localhost``, and fail-closed on unresolved hosts;
  * the media variant's loopback allowance (clients hand the server's own
    media URLs back);
  * ``INGEST_ALLOW_HOSTS`` authority (listed ⇒ allowed despite private,
    unlisted ⇒ rejected) and the legacy disable switch;
  * the pin itself: single resolution (the TOCTOU property, asserted by
    counting getaddrinfo calls), IP-rewritten URL, Host/SNI preservation,
    port handling, literal-IP and proxy-mode pin inactivity;
  * fetch-point wiring: ``_download_url`` (ingest), ``_afetch_media_bytes``
    (embed/query), ``_probe_remote_media_type`` (classification) against a
    real loopback HTTP server — including redirect hops re-validated
    (a redirect into private/metadata space is refused).

All offline: loopback servers + monkeypatched resolvers only.

Run::

    pytest tests/security/test_url_policy.py -q
"""

import asyncio
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.utils import url_policy as up


@pytest.fixture(autouse=True)
def _clean_policy_env(monkeypatch):
    """Every test starts from the default policy (no allowlist, block on),
    regardless of the ambient environment."""
    monkeypatch.setattr(up, "_INGEST_ALLOW_HOSTS", ())
    monkeypatch.setattr(up, "_INGEST_BLOCK_PRIVATE", True)
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    yield


# ---------------------------------------------------------------------------
# Deny classes — literal IPs (no DNS involved)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://127.8.8.8/x",
        "http://10.0.0.1/x",
        "http://172.16.0.1/x",
        "http://172.31.255.255/x",
        "http://192.168.1.100/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0/x",
        "http://[::1]/x",
        "http://[fe80::1]/x",
        "http://[fc00::abcd]/x",
        "http://224.0.0.1/x",
        "http://240.0.0.1/x",
    ],
)
def test_ingest_policy_blocks_private_literals(url):
    with pytest.raises(ValueError):
        up._check_url_policy(url)
    with pytest.raises(ValueError):
        up.validate_fetch_url(url, allow_loopback=False)


@pytest.mark.parametrize("url", ["http://8.8.8.8/x", "http://93.184.216.34/file.jpg"])
def test_ingest_policy_allows_public_literals(url):
    up._check_url_policy(url)  # must not raise


def test_media_policy_allows_loopback_blocks_rest():
    # clients legitimately hand back the server's own media URLs
    up._check_media_url_policy("http://localhost:8000/api/datasets/d/files/a.jpg?token=t")
    up._check_media_url_policy("http://127.0.0.1:8000/x.jpg")
    up._check_media_url_policy("http://[::1]:8000/x.jpg")
    with pytest.raises(ValueError):
        up._check_media_url_policy("http://10.1.2.3/x.jpg")
    with pytest.raises(ValueError):
        up._check_media_url_policy("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(ValueError):
        up._check_media_url_policy("http://192.168.0.44/x.jpg")


def test_non_http_schemes_are_not_url_policy_business():
    # file:// (media_paths), s3:// (dedicated S3 paths), data: — inert here
    up._check_url_policy("file:///etc/passwd")
    up._check_media_url_policy("s3://bucket/key")
    up._check_url_policy("ftp://example.com/x")


# ---------------------------------------------------------------------------
# Deny classes — hostname resolution (monkeypatched)
# ---------------------------------------------------------------------------


@pytest.fixture
def resolver(monkeypatch):
    """Replace socket.getaddrinfo with a controllable table."""
    table = {}

    def _getaddrinfo(host, port=None, *args, **kwargs):
        if host in table:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], port or 0))]
        raise socket.gaierror(-2, f"Name or service not known ({host})")

    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)
    return table


def test_public_hostname_allowed(resolver):
    resolver["cdn.example.com"] = "93.184.216.34"
    up._check_url_policy("https://cdn.example.com/a.jpg")


def test_private_resolving_hostname_blocked(resolver):
    resolver["rebind.example.com"] = "10.9.8.7"
    with pytest.raises(ValueError):
        up._check_url_policy("https://rebind.example.com/a.jpg")
    with pytest.raises(ValueError):
        up._check_media_url_policy("https://rebind.example.com/a.jpg")


def test_unresolved_hostname_fail_closed(resolver):
    with pytest.raises(ValueError):
        up._check_url_policy("https://does-not-resolve.invalid/a.jpg")


def test_localhost_blocked_for_ingest(resolver):
    with pytest.raises(ValueError):
        up._check_url_policy("http://localhost/admin")


# ---------------------------------------------------------------------------
# Allowlist authority + disable switch
# ---------------------------------------------------------------------------


def test_allowlist_is_authoritative(resolver):
    resolver["minio.internal.svc.cluster.local"] = "10.1.2.3"
    resolver["other.example.com"] = "93.184.216.34"
    up._INGEST_ALLOW_HOSTS = (".internal.svc.cluster.local",)
    # listed: allowed despite resolving privately (in-cluster ingestion)
    up._check_url_policy("http://minio.internal.svc.cluster.local/bucket/f.jpg")
    # unlisted: rejected EVEN when it resolves publicly
    with pytest.raises(ValueError, match="INGEST_ALLOW_HOSTS"):
        up._check_url_policy("http://other.example.com/f.jpg")
    # the media variant keeps the same authority
    up._check_media_url_policy("http://minio.internal.svc.cluster.local/bucket/f.jpg")


def test_block_private_disable_restores_legacy(resolver):
    resolver["rebind.example.com"] = "10.9.8.7"
    up._INGEST_BLOCK_PRIVATE = False
    up._check_url_policy("https://rebind.example.com/a.jpg")  # legacy permissive


# ---------------------------------------------------------------------------
# The DNS-rebinding pin
# ---------------------------------------------------------------------------


def test_pin_rewrites_host_to_validated_ip(resolver):
    resolver["cdn.example.com"] = "93.184.216.34"
    p = up.validate_fetch_url("https://cdn.example.com/some/path.img?q=1")
    assert p.pin_active is True
    assert p.host == "cdn.example.com" and p.ip == "93.184.216.34"
    assert p.pinned_url == "https://93.184.216.34/some/path.img?q=1"
    assert p.host_header == "cdn.example.com"  # virtual-host routing intact
    assert p.sni_hostname == "cdn.example.com"  # TLS cert name intact
    assert p.url == "https://cdn.example.com/some/path.img?q=1"


def test_pin_preserves_explicit_port(resolver):
    resolver["svc.example.com"] = "93.184.216.34"
    p = up.validate_fetch_url("http://svc.example.com:8443/f.bin")
    assert p.port == 8443
    assert p.host_header == "svc.example.com:8443"
    assert p.pinned_url == "http://93.184.216.34:8443/f.bin"
    assert p.sni_hostname == ""  # no TLS SNI on http


def test_pin_default_ports(resolver):
    resolver["a.example.com"] = "93.184.216.34"
    resolver["b.example.com"] = "93.184.216.34"
    assert up.validate_fetch_url("http://a.example.com/x").host_header == "a.example.com"
    assert up.validate_fetch_url("https://b.example.com/x").host_header == "b.example.com"


def test_pin_single_resolution_toctou_property(resolver, monkeypatch):
    """Check-time and fetch-time DNS must be ONE resolution: classification
    and the pin judge the same address set."""
    calls = []

    real = socket.getaddrinfo

    def _counting(host, port=None, *a, **k):
        calls.append(host)
        return (
            real(host, port, *a, **k)
            if host in ("nope",)
            else [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 0))]
        )

    monkeypatch.setattr(socket, "getaddrinfo", _counting)
    p = up.validate_fetch_url("http://one-resolution.example/f.jpg")
    assert p.pin_active and p.ip == "93.184.216.34"
    assert len(calls) == 1, f"expected exactly ONE resolution, got {len(calls)}"


def test_pin_refuses_private_resolution(resolver):
    """A rebinding answer (private IP for a public name) is refused — the
    fetch never sees it."""
    resolver["evil.example.com"] = "192.168.0.9"
    with pytest.raises(ValueError):
        up.validate_fetch_url("http://evil.example.com/f.jpg")


def test_pin_ipv6_answer_prefers_ipv4(resolver, monkeypatch):
    def _dual(host, port=None, *a, **k):
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800:220:1:248:1893:25c8:1946", port or 0, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", _dual)
    p = up.validate_fetch_url("http://dual.example/f.jpg")
    assert p.ip == "93.184.216.34", "an IPv4 answer wins (pods commonly lack v6 routes)"


def test_pin_inactive_for_literal_ip():
    p = up.validate_fetch_url("http://93.184.216.34/f.jpg")
    assert p.pin_active is False and p.pinned_url == "http://93.184.216.34/f.jpg"


def test_pin_inactive_under_proxy(monkeypatch, resolver):
    """Proxy residual (documented, same as the searxng port): egress DNS
    belongs to the proxy; the pin is skipped, the policy still ran."""
    resolver["cdn.example.com"] = "93.184.216.34"
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
    p = up.validate_fetch_url("https://cdn.example.com/x.jpg")
    assert p.pin_active is False
    assert p.pinned_url == "https://cdn.example.com/x.jpg"


def test_pin_under_proxy_still_classifies(monkeypatch, resolver):
    resolver["rebind.example.com"] = "10.0.0.9"
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")
    with pytest.raises(ValueError):
        up.validate_fetch_url("http://rebind.example.com/f.jpg")


def test_pin_rejects_unresolved_and_bad_schemes():
    with pytest.raises(ValueError):
        up.validate_fetch_url("http://does-not-resolve.invalid/x")
    with pytest.raises(ValueError, match="scheme"):
        up.validate_fetch_url("gopher://example.com/x")
    with pytest.raises(ValueError):
        up.validate_fetch_url("http://")  # no host


# ---------------------------------------------------------------------------
# Fetch-point wiring against a REAL loopback server
# ---------------------------------------------------------------------------


_JPEG = b"\xff\xd8\xff\xe0" + b"JFIF-payload" * 8


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/direct.jpg":
            self._serve(_JPEG, "image/jpeg")
        elif self.path == "/redirect-ok":
            self._redirect("/direct.jpg")
        elif self.path == "/redirect-private":
            self._redirect("http://10.9.9.9/x.jpg")
        elif self.path == "/redirect-metadata":
            self._redirect("http://169.254.169.254/latest/meta-data/")
        elif self.path == "/redirect-loop":
            self._redirect("/redirect-loop")
        elif self.path == "/download.bin":
            self._serve(b"DOWNLOADBODY" * 10, "application/octet-stream")
        else:
            self.send_response(404)
            self.end_headers()

    def _serve(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def loopback_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def test_media_fetch_pins_and_follows_validated_redirects(loopback_server):
    from multimodal_rag.rag_system import _afetch_media_bytes

    assert asyncio.run(_afetch_media_bytes(f"{loopback_server}/direct.jpg")) == _JPEG
    # a redirect is just another remote fetch: re-validated + pinned
    assert asyncio.run(_afetch_media_bytes(f"{loopback_server}/redirect-ok")) == _JPEG


@pytest.mark.parametrize("path", ["/redirect-private", "/redirect-metadata"])
def test_media_fetch_refuses_redirect_into_private_space(loopback_server, path):
    """Blind follow_redirects never re-checked a hop's target — a public
    media URL could bounce the fetch into private/metadata space.  Each hop
    is now validated (the same hole the searxng port closed)."""
    from multimodal_rag.rag_system import _afetch_media_bytes

    with pytest.raises(ValueError):
        asyncio.run(_afetch_media_bytes(f"{loopback_server}{path}"))


def test_media_fetch_redirect_cap(loopback_server, monkeypatch):
    import multimodal_rag.rag_system as rs

    monkeypatch.setattr(rs, "_MEDIA_MAX_REDIRECTS", 3)
    from multimodal_rag.rag_system import _afetch_media_bytes

    with pytest.raises(ValueError, match="redirect"):
        asyncio.run(_afetch_media_bytes(f"{loopback_server}/redirect-loop"))


def test_ingest_download_refuses_loopback(loopback_server):
    """Ingest-time policy blocks loopback like every private range — the
    guard holds on the pinned path."""
    from multimodal_rag.dataset_manager import _download_url

    with pytest.raises(ValueError):
        _download_url(f"{loopback_server}/download.bin")


def test_ingest_download_end_to_end_via_allowlisted_private_target(loopback_server, monkeypatch):
    """The in-cluster ingestion pattern (allowlisted private target) through
    the REAL pinned download path: body lands, redirects re-validated."""
    from multimodal_rag import dataset_manager as dm
    from multimodal_rag.utils import url_policy as up

    real = socket.getaddrinfo
    host = "minio.pin-test.example"
    port = loopback_server.rsplit(":", 1)[-1]

    def _getaddrinfo(h, p=None, *a, **k):
        if h == host:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", p or 0))]
        return real(h, p, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)
    up._INGEST_ALLOW_HOSTS = (".pin-test.example",)
    try:
        out = dm._download_url(f"http://{host}:{port}/download.bin")
        with open(out, "rb") as f:
            assert f.read() == b"DOWNLOADBODY" * 10
        os.unlink(out)
        # redirect hop re-validated through the pin
        out = dm._download_url(f"http://{host}:{port}/redirect-ok")
        os.unlink(out)
    finally:
        up._INGEST_ALLOW_HOSTS = ()


def test_probe_classifies_and_refuses_private(loopback_server):
    from multimodal_rag.mcp_server import _probe_remote_media_type

    assert _probe_remote_media_type(f"{loopback_server}/direct.jpg") == "image"
    # redirect followed through the pin
    assert _probe_remote_media_type(f"{loopback_server}/redirect-ok") == "image"
    # a private literal is refused INSIDE the probe (best-effort → None,
    # no fetch ever happened)
    assert _probe_remote_media_type("http://10.1.2.3/x.jpg") is None
    assert _probe_remote_media_type(f"{loopback_server}/redirect-private") is None
