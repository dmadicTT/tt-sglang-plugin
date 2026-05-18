"""Tenstorrent platform plugin for SGLang."""

from __future__ import annotations

import os


def activate() -> str | None:
    """Return the Tenstorrent SRT platform class when the plugin is enabled."""
    if os.getenv("SGLANG_TENSTORRENT_MOCK") != "1":
        return None
    return "sglang_tenstorrent.platform.TenstorrentPlatform"


def register() -> None:
    """Register Tenstorrent plugin model implementations."""
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang_tenstorrent.sampler import register as register_sampler

    ModelRegistry.register("sglang_tenstorrent.models", overwrite=True)
    register_sampler()
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
        # CPU's __post_init__ branch hard-resets sampling_backend to "pytorch"
        # (server_args.py:1161-1165), undoing any --sampling-backend flag.
        # Re-apply tenstorrent so the mock isn't taxed by the full host sampler.
        server_args.sampling_backend = "tenstorrent"
    return result
