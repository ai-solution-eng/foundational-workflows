"""MINIATURE fixture of the gateway distiller tests (patcher tests only)."""

from pcai_llm.core import distiller as dist_mod


def test_prompt_version_is_set() -> None:
    assert dist_mod.PROMPT_VERSION == "dsh-2026-10-09"


def test_prompt_carries_index_contract() -> None:
    assert dist_mod.DISTILL_SYSTEM_PROMPT.startswith("You distill AI-coding-assistant")
