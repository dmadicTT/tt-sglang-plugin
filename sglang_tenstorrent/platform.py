"""SRT platform stub for Tenstorrent hardware."""

import psutil

from sglang.srt.platforms.interface import SRTPlatform
from sglang_tenstorrent.device import TenstorrentDeviceMixin
from sglang_tenstorrent.deepseek_r1_0528 import (
    DEEPSEEK_R1_0528_MODEL_ID,
    is_deepseek_r1_0528_mock_model_path,
)


class TenstorrentPlatform(SRTPlatform, TenstorrentDeviceMixin):
    def apply_server_args_defaults(self, server_args) -> None:
        if is_deepseek_r1_0528_mock_model_path(server_args.model_path):
            server_args.served_model_name = DEEPSEEK_R1_0528_MODEL_ID

    def get_device_total_memory(self, device_id: int = 0) -> int:
        del device_id
        return psutil.virtual_memory().total

    def get_current_memory_usage(self, device=None) -> float:
        del device
        memory = psutil.virtual_memory()
        return float(memory.total - memory.available)

    def get_default_attention_backend(self) -> str:
        return "torch_native"

    def get_mha_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool

    def get_paged_allocator_cls(self) -> type:
        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

        return PagedTokenToKVPoolAllocator
