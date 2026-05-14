"""Tenstorrent platform plugin for SGLang."""

from __future__ import annotations

import os
from importlib.util import find_spec


def activate() -> str | None:
    """Return the Tenstorrent SRT platform class when TT-NN is available."""
    if find_spec("ttnn") is None and os.getenv("SGLANG_TENSTORRENT_MOCK") != "1":
        return None
    return "sglang_tenstorrent.platform.TenstorrentPlatform"


def register() -> None:
    """Register Tenstorrent plugin model implementations."""
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    ModelRegistry.register("sglang_tenstorrent.models", overwrite=True)
    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.__post_init__",
        _apply_mock_model_identity,
        HookType.AFTER,
    )


def _apply_mock_model_identity(result, server_args):
    from sglang_tenstorrent.deepseek_r1_0528 import (
        DEEPSEEK_R1_0528_MODEL_ID,
        is_deepseek_r1_0528_mock_model_path,
    )

    if is_deepseek_r1_0528_mock_model_path(server_args.model_path):
        server_args.served_model_name = DEEPSEEK_R1_0528_MODEL_ID
    return result
