"""Per-request mm_processor_kwargs are opt-in (the vLLM >= 0.31 trust gate).

vLLM answers 400 on ANY request carrying mm_processor_kwargs unless the server
was started with --trust-request-mm-kwargs:

    400: Per-request mm_processor_kwargs are disabled by default because they
    can change multimodal preprocessing resource usage.

So the field must be ABSENT by default.  When a deployment opts in, media this
process already normalized (tier-3 data URLs, client-sampled frames) drops the
keys the client enforced itself and keeps only what the server can still act on.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from multimodal_rag.utils.model_adapters import (
    MultiModalEmbeddings,
    MultiModalReranker,
    _mm_processor_kwargs_payload,
)

MM_KWARGS: dict[str, Any] = {
    "fps": 1.0,
    "max_frames": 64,
    "min_pixels": 4096,
    "max_pixels": 720 * 720,
    "total_pixels": 5 * 720 * 720,
}

# Enforced client-side for pre-normalized media: _resize_image caps the pixels,
# _fetch_video_frames pre-samples the frames.  min_pixels is NOT here -- the
# client never upscales, so only the server can apply that floor.
CLIENT_ENFORCED = {"max_pixels", "total_pixels", "fps", "max_frames"}

TIER3_IMAGE = "data:image/png;base64,AAAA"
REMOTE_IMAGE = "https://example.com/page.png"
VIDEO = "data:video/mp4;base64,AAAA"

ENV = "MODEL_SEND_MM_PROCESSOR_KWARGS"


class _Recorder:
    """Stands in for the OpenAI async client, capturing every request body."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.paths: list[str] = []
        self.bodies: list[dict[str, Any]] = []

    async def post(self, path: str, **kwargs: Any) -> Any:
        self.paths.append(path)
        self.bodies.append(kwargs["body"])
        return self.response


def _msg(part_type: str, url: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": part_type, part_type: {"url": url}}]}


TEXT_MSG: dict[str, Any] = {"role": "user", "content": [{"type": "text", "text": "hello"}]}


def _embedder() -> tuple[MultiModalEmbeddings, _Recorder]:
    rec = _Recorder(SimpleNamespace(data=[SimpleNamespace(embedding=[0.0, 1.0])]))
    model = SimpleNamespace(model_name="test-model", async_client=rec, mm_processor_kwargs=MM_KWARGS)
    adapter = MultiModalEmbeddings.__new__(MultiModalEmbeddings)
    adapter.emb = model
    return adapter, rec


def _reranker() -> tuple[MultiModalReranker, _Recorder]:
    rec = _Recorder({"data": [{"score": 0.5}], "results": [{"index": 0, "relevance_score": 0.5}]})
    model = SimpleNamespace(model_name="test-reranker", async_client=rec, mm_processor_kwargs=MM_KWARGS)
    adapter = MultiModalReranker.__new__(MultiModalReranker)
    adapter.emb = model
    return adapter, rec


def _embed(adapter: MultiModalEmbeddings, messages: list[dict[str, Any]]) -> dict[str, Any]:
    asyncio.run(adapter._embed_single_message_unbounded(messages))
    return adapter.emb.async_client.bodies[0]  # type: ignore[attr-defined]


def _score(adapter: MultiModalReranker, query: dict[str, Any], docs: list[dict[str, Any]]) -> dict[str, Any]:
    asyncio.run(adapter._score_one_query(query, docs))
    return adapter.emb.async_client.bodies[0]  # type: ignore[attr-defined]


def _rerank(adapter: MultiModalReranker, query: dict[str, Any], docs: list[dict[str, Any]]) -> dict[str, Any]:
    asyncio.run(adapter._rerank_one_query(query, docs))
    return adapter.emb.async_client.bodies[0]  # type: ignore[attr-defined]


# -- the flag itself ---------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_flag_truthy_values_enable_the_field(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ENV, value)
    assert _mm_processor_kwargs_payload(MM_KWARGS, [_msg("image_url", REMOTE_IMAGE)])


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no", "nope"])
def test_flag_falsy_values_disable_the_field(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ENV, value)
    assert _mm_processor_kwargs_payload(MM_KWARGS, [_msg("image_url", REMOTE_IMAGE)]) == {}


def test_flag_unset_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    assert _mm_processor_kwargs_payload(MM_KWARGS, [_msg("image_url", REMOTE_IMAGE)]) == {}


# -- /embeddings -------------------------------------------------------------


def test_embed_body_omits_the_field_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression under test: a hardened vLLM 400s on any body carrying it."""
    monkeypatch.delenv(ENV, raising=False)
    adapter, _ = _embedder()
    body = _embed(adapter, [_msg("image_url", TIER3_IMAGE)])

    assert "mm_processor_kwargs" not in body
    # The rest of the request is untouched.
    assert body["model"] == "test-model"
    assert body["messages"] == [_msg("image_url", TIER3_IMAGE)]
    assert body["encoding_format"] == "float"


def test_embed_body_omits_the_field_when_opted_in_but_kwargs_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV, "1")
    rec = _Recorder(SimpleNamespace(data=[SimpleNamespace(embedding=[0.0])]))
    model = SimpleNamespace(model_name="m", async_client=rec, mm_processor_kwargs={})
    adapter = MultiModalEmbeddings.__new__(MultiModalEmbeddings)
    adapter.emb = model

    body = _embed(adapter, [_msg("image_url", REMOTE_IMAGE)])
    assert "mm_processor_kwargs" not in body


def test_embed_tier3_image_drops_client_enforced_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tier-3 media is already sized and sampled here -- only the floor survives."""
    monkeypatch.setenv(ENV, "1")
    adapter, _ = _embedder()
    body = _embed(adapter, [_msg("image_url", TIER3_IMAGE)])

    sent = body["mm_processor_kwargs"]
    assert set(sent) == {"min_pixels"}
    assert not CLIENT_ENFORCED & set(sent)


def test_embed_remote_image_keeps_the_full_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server fetches a remote image, so its own budget still applies."""
    monkeypatch.setenv(ENV, "1")
    adapter, _ = _embedder()
    body = _embed(adapter, [_msg("image_url", REMOTE_IMAGE)])

    assert body["mm_processor_kwargs"] == MM_KWARGS


def test_embed_video_keeps_the_full_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A video_url means the server still samples -- fps/max_frames are live."""
    monkeypatch.setenv(ENV, "1")
    adapter, _ = _embedder()
    body = _embed(adapter, [_msg("video_url", VIDEO)])

    assert body["mm_processor_kwargs"] == MM_KWARGS


def test_embed_text_only_never_sends_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """No media, nothing for a processor budget to do -- even when opted in."""
    monkeypatch.setenv(ENV, "1")
    adapter, _ = _embedder()
    body = _embed(adapter, [TEXT_MSG])

    assert "mm_processor_kwargs" not in body


def test_embed_mixed_media_falls_back_to_the_full_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """One server-side part is enough to keep the whole budget."""
    monkeypatch.setenv(ENV, "1")
    adapter, _ = _embedder()
    body = _embed(
        adapter,
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": TIER3_IMAGE}},
                    {"type": "video_url", "video_url": {"url": VIDEO}},
                ],
            }
        ],
    )

    assert body["mm_processor_kwargs"] == MM_KWARGS


# -- /score and /rerank ------------------------------------------------------


def test_score_body_omits_the_field_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    adapter, _ = _reranker()
    body = _score(adapter, _msg("image_url", TIER3_IMAGE), [_msg("image_url", TIER3_IMAGE)])

    assert "mm_processor_kwargs" not in body
    assert body["text_2"]


def test_rerank_body_omits_the_field_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    adapter, _ = _reranker()
    body = _rerank(adapter, _msg("image_url", TIER3_IMAGE), [_msg("image_url", TIER3_IMAGE)])

    assert "mm_processor_kwargs" not in body
    assert body["documents"]


def test_score_tier3_drops_client_enforced_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV, "1")
    adapter, _ = _reranker()
    body = _score(adapter, _msg("image_url", TIER3_IMAGE), [_msg("image_url", TIER3_IMAGE)])

    assert set(body["mm_processor_kwargs"]) == {"min_pixels"}


def test_rerank_remote_media_keeps_the_full_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV, "1")
    adapter, _ = _reranker()
    body = _rerank(adapter, _msg("image_url", REMOTE_IMAGE), [_msg("image_url", REMOTE_IMAGE)])

    assert body["mm_processor_kwargs"] == MM_KWARGS


def test_score_text_only_never_sends_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV, "1")
    adapter, _ = _reranker()
    body = _score(adapter, TEXT_MSG, [TEXT_MSG])

    assert "mm_processor_kwargs" not in body


# -- helper contract ---------------------------------------------------------


def test_helper_ignores_non_dict_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """_remove_extra_input_components flattens text docs to plain strings."""
    monkeypatch.setenv(ENV, "1")
    assert _mm_processor_kwargs_payload(MM_KWARGS, ["plain text doc"]) == {}
