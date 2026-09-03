"""Shared URL-fetch policy (SSRF guards) for remote http(s) fetches.

Used by:

* ``dataset_manager._download_url`` — ingest-time downloads of user-supplied
  URLs (``batch-urls``, ``add_file`` with a URL);
* ``rag_system`` — media refs inside user-supplied documents (``image`` /
  ``video`` / ``audio`` keys), fetched at embed time and query time;
* ``mcp_server`` — query-time media URLs on the MCP tools.

Lives in ``utils/`` (not ``dataset_manager``) so ``rag_system`` can import it
without a circular import (``dataset_manager`` imports ``rag_system`` at
module level).  ``dataset_manager`` re-exports the public names so existing
``from multimodal_rag.dataset_manager import _check_media_url_policy`` call
sites keep working.

Configuration (read once at import):

* ``INGEST_ALLOW_HOSTS`` — comma-separated host allowlist.  An entry like
  ``.minio.svc.cluster.local`` matches the zone and subdomains.  When set it
  is authoritative: listed hosts are allowed (even when they resolve
  privately — that is how in-cluster MinIO/internal ingestions are
  permitted) and everything else is rejected.
* ``INGEST_BLOCK_PRIVATE_HOSTS`` — private/loopback/link-local block toggle,
  default on.  ``false`` restores the legacy permissive behaviour.
"""

import os

from multimodal_rag.utils.logging_utils import logging

logger = logging.getLogger(__name__)

# Optional comma-separated allowlist of hosts for http(s) fetches.  Empty
# = all hosts allowed (subject to the private-range block below).
_INGEST_ALLOW_HOSTS = tuple(
    h.strip().lower() for h in os.environ.get("INGEST_ALLOW_HOSTS", "").split(",") if h.strip()
)

# Private-range block (DNS + literal-IP).  On by default so remote fetches
# cannot reach internal/loopback targets (SSRF).
_INGEST_BLOCK_PRIVATE = os.environ.get("INGEST_BLOCK_PRIVATE_HOSTS", "true").lower() in ("1", "true", "yes")


def _host_matches_allowlist(host: str) -> bool:
    host = host.lower()
    for pat in _INGEST_ALLOW_HOSTS:
        if pat.startswith("."):
            if host == pat[1:] or host.endswith(pat):
                return True
        elif host == pat:
            return True
    return False


def _host_is_private(host: str, allow_loopback: bool = False) -> bool:
    """Return True if *host* is or resolves to a private/loopback/link-local address.

    With ``allow_loopback=True`` (the query-time media policy) loopback
    addresses and the literal name ``localhost`` are *not* considered
    private: clients legitimately hand the server's own media URLs
    (``http://localhost:8000/api/datasets/...``) back to the query tools.
    """
    import ipaddress
    import socket

    hostname = host.rsplit(":", 1)[0].strip("[]")
    if hostname == "localhost":
        return not allow_loopback
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None
    if ip is not None:
        if ip.is_loopback and allow_loopback:
            return False
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True  # unresolved — safest to treat as suspicious when blocking is on
    for info in addrinfos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if ip.is_loopback and allow_loopback:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return True
    return False


def _check_url_policy(url: str) -> None:
    """Raise :class:`ValueError` if *url* violates the configured URL policy.

    Ingest-time variant: loopback is blocked like every other private range
    (a remote ingest URL has no business pointing at the server itself).
    """
    if not url.startswith(("http://", "https://")):
        return
    from urllib.parse import urlparse

    host = urlparse(url).hostname or ""
    # An explicit allowlist is authoritative: hosts not listed are rejected,
    # and listed hosts are allowed even when they resolve privately (that is
    # how in-cluster MinIO/internal ingestions are permitted).
    if _INGEST_ALLOW_HOSTS:
        if not _host_matches_allowlist(host):
            raise ValueError(
                f"URL host '{host}' is not allowed by INGEST_ALLOW_HOSTS"
                + (f"={','.join(_INGEST_ALLOW_HOSTS)}" if _INGEST_ALLOW_HOSTS else "")
            )
        return
    if _INGEST_BLOCK_PRIVATE and _host_is_private(host):
        raise ValueError(f"URL host '{host}' resolves to a private/internal address (INGEST_BLOCK_PRIVATE_HOSTS=true)")


def _check_media_url_policy(url: str) -> None:
    """Policy for *media* fetches of user-supplied http(s) refs.

    Covers both query-time media URLs (search with image/video/audio,
    ``describe_media``, ``transcribe_audio``) and media refs embedded in
    user-supplied documents (``POST /documents``, MCP ``add_memory``), which
    the server fetches at embed time and again at query time.

    Same rules as :func:`_check_url_policy` with one difference: loopback is
    allowed by default, because clients legitimately pass the server's own
    media URLs (``http://localhost:8000/api/datasets/...?token=...``) back
    to these tools.  ``INGEST_ALLOW_HOSTS`` remains authoritative when set;
    set ``INGEST_BLOCK_PRIVATE_HOSTS=false`` to disable (not recommended).
    """
    if not url.startswith(("http://", "https://")):
        return
    from urllib.parse import urlparse

    host = urlparse(url).hostname or ""
    if _INGEST_ALLOW_HOSTS:
        if not _host_matches_allowlist(host):
            raise ValueError(
                f"URL host '{host}' is not allowed by INGEST_ALLOW_HOSTS"
                + (f"={','.join(_INGEST_ALLOW_HOSTS)}" if _INGEST_ALLOW_HOSTS else "")
            )
        return
    if _INGEST_BLOCK_PRIVATE and _host_is_private(host, allow_loopback=True):
        raise ValueError(
            f"URL host '{host}' resolves to a private/internal address "
            f"(INGEST_BLOCK_PRIVATE_HOSTS=true; add it to INGEST_ALLOW_HOSTS to permit)"
        )
