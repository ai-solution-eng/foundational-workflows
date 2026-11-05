"""MCP ``inspect_media`` — modality routing (the two former single-modality tools merged).

The merge (2026-11) replaced two single-modality tools with one that routes on
the media ref:

* image/video → the VLM description path (same body as the former VLM tool);
* audio → the ASR transcript path (same body as the former ASR tool);
* ambiguous refs → a ToolError that ASKS for ``media_type`` (never a guess
  that silently sends audio to the VLM);
* ``query`` on audio and ``max_seconds`` on image/video → explicit errors
  rather than silently ignored parameters.

The assertions run through ``mcp.call_tool` (what a client actually calls) and
stop at the model check — no VLM/ASR is configured in the stub manager, so the
"which model was selected" outcome IS the routing proof.

Run::

    pytest tests/security/test_inspect_media_routing.py -q
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.utils.media_paths as mp

_JPEG = b"\xff\xd8\xff" + b"\x00" * 16
_FTYP = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 12
_ID3 = b"ID3\x04\x00" + b"\x00" * 16


@pytest.fixture()
def mcp_env(monkeypatch, tmp_path):
    """The MCP module with a stub manager and a staging tree of media refs.

    ``vlm`` and ``asr`` are both ``None``: the tool raises its
    "No VLM/ASR model configured" error immediately after routing, which is
    exactly the signal these tests read.
    """
    import multimodal_rag.mcp_server as mcp

    staging = tmp_path / "staging"
    staging.mkdir()
    files = {
        "shot.jpg": _JPEG,
        "clip.mp4": _FTYP,
        "talk.mp3": _ID3,
        "noext_audio": _ID3,
        "noext_video": _FTYP,
        "blob": b"plain bytes, no signature",
    }
    for name, blob in files.items():
        (staging / name).write_bytes(blob)
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_PATH_PREFIXES", (str(staging),))
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_ANY", False)

    class _StubDM:
        vlm = None
        asr = None

        class embedder:  # attribute-style stub
            mm_processor_kwargs = {}  # noqa: RUF012 — attribute-style stub

    monkeypatch.setattr(mcp, "get_manager", lambda: _StubDM())
    return mcp, staging


def _inspect(mcp, **args):
    """Call the tool the way a client does; return (status, text)."""
    import multimodal_rag.mcp_server as mcp_module

    async def _run():
        try:
            result = await mcp.mcp.call_tool("inspect_media", args)
            if getattr(result, "isError", False):
                return ("error", str(result.content[0].text))
            return ("ok", str(result.content[0].text))
        except mcp_module.ToolError as exc:
            return ("error", str(exc))

    return asyncio.run(_run())


# ===========================================================================
# 1 · routing: extension / magic bytes → the right model
# ===========================================================================


@pytest.mark.parametrize(
    ("name", "model_error"),
    [
        ("shot.jpg", "No VLM model configured"),  # image extension
        ("clip.mp4", "No VLM model configured"),  # video extension
        ("talk.mp3", "No ASR model configured"),  # audio extension
        ("noext_audio", "No ASR model configured"),  # audio magic bytes (ID3)
        ("noext_video", "No VLM model configured"),  # video magic bytes (ftyp)
    ],
)
def test_ref_is_routed_to_the_right_model(mcp_env, name, model_error):
    mcp, staging = mcp_env
    status, text = _inspect(mcp, media_url=f"file://{(staging / name).resolve()}")
    assert status == "error", text
    assert model_error in text


def test_data_url_mime_routing(mcp_env):
    mcp, _staging = mcp_env
    _s, text = _inspect(mcp, media_url="data:audio/mpeg;base64,AAAA")
    assert "No ASR model configured" in text
    _s, text = _inspect(mcp, media_url="data:image/png;base64,AAAA")
    assert "No VLM model configured" in text


def test_explicit_media_type_wins_over_the_ref(mcp_env):
    """An extensionless ref is routable by hint alone — the documented escape
    hatch for ambiguous URLs."""
    mcp, staging = mcp_env
    ref = f"file://{(staging / 'blob').resolve()}"
    _s, text = _inspect(mcp, media_url=ref, media_type="audio")
    assert "No ASR model configured" in text
    _s, text = _inspect(mcp, media_url=ref, media_type="video")
    assert "No VLM model configured" in text


# ===========================================================================
# 2 · ambiguity and parameter validation
# ===========================================================================


def test_ambiguous_ref_asks_for_media_type(mcp_env):
    """No extension, no sniffable signature → the caller is TOLD what to pass;
    the tool never guesses a modality."""
    mcp, staging = mcp_env
    status, text = _inspect(mcp, media_url=f"file://{(staging / 'blob').resolve()}")
    assert status == "error"
    assert "Could not determine" in text
    assert "media_type='image'" in text and "media_type='video'" in text and "media_type='audio'" in text


def test_invalid_media_type_rejected(mcp_env):
    mcp, staging = mcp_env
    status, text = _inspect(mcp, media_url=f"file://{(staging / 'shot.jpg').resolve()}", media_type="gif")
    assert status == "error"
    assert "'media_type' must be 'image', 'video' or 'audio'" in text


def test_query_rejected_for_audio(mcp_env):
    """The ASR model cannot answer questions — refuse instead of ignoring it."""
    mcp, staging = mcp_env
    status, text = _inspect(mcp, media_url=f"file://{(staging / 'talk.mp3').resolve()}", query="what did they say?")
    assert status == "error"
    assert "'query' is not supported for audio" in text


def test_empty_query_is_not_a_rejection(mcp_env):
    """An empty/blank query is "not provided" — it must not trip the audio guard."""
    mcp, staging = mcp_env
    _s, text = _inspect(mcp, media_url=f"file://{(staging / 'talk.mp3').resolve()}", query="")
    assert "No ASR model configured" in text


def test_max_seconds_rejected_for_image_and_video(mcp_env):
    """max_seconds is audio-only — refuse it rather than silently ignoring it."""
    mcp, staging = mcp_env
    for name in ("shot.jpg", "clip.mp4"):
        status, text = _inspect(mcp, media_url=f"file://{(staging / name).resolve()}", max_seconds=5.0)
        assert status == "error", text
        assert "'max_seconds' only applies to audio" in text


def test_max_seconds_accepted_for_audio(mcp_env):
    mcp, staging = mcp_env
    _s, text = _inspect(mcp, media_url=f"file://{(staging / 'talk.mp3').resolve()}", max_seconds=5.0)
    assert "No ASR model configured" in text
