"""DeepSeek R1 0528 mock model for Tenstorrent plugin bring-up."""

from __future__ import annotations

import os
import time
from typing import Iterable

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

MOCK_TSU_ENV = "SGLANG_TENSTORRENT_MOCK_TSU"


def _resolve_mock_tsu(config) -> float:
    env_value = os.environ.get(MOCK_TSU_ENV)
    raw = env_value if env_value else getattr(config, "mock_tsu", 500.0)
    tsu = float(raw)
    if tsu <= 0:
        raise ValueError(
            f"mock_tsu must be > 0 (got {tsu!r}); set {MOCK_TSU_ENV} or mock_tsu in config.json"
        )
    return tsu


class TenstorrentDeepSeekR10528ForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, **_kwargs):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = getattr(config, "vocab_size", 256)
        self.eos_token_id = getattr(config, "eos_token_id", 2)
        self.max_generated_tokens = getattr(config, "mock_max_generated_tokens", 16)
        self.mock_tsu = _resolve_mock_tsu(config)
        self.token_delay_seconds = 1.0 / self.mock_tsu
        self._generated_by_request: dict[str | int, int] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        del input_ids, positions

        time.sleep(self.token_delay_seconds)
        req_pool_indices = forward_batch.req_pool_indices.detach().cpu().tolist()
        request_keys = forward_batch.rids or req_pool_indices
        logits = torch.full(
            (len(req_pool_indices), self.vocab_size),
            -1.0e9,
            dtype=torch.float32,
            device=forward_batch.seq_lens.device,
        )

        for row, request_key in enumerate(request_keys):
            generated = self._generated_by_request.get(request_key, 0)
            token_id = self._sample_next_token_on_device(generated)
            if token_id != self.eos_token_id:
                self._generated_by_request[request_key] = generated + 1
            # Compatibility shim: SGLang currently samples from logits, while
            # Tenstorrent will already have sampled token IDs on device.
            logits[row, token_id] = 0.0

        return LogitsProcessorOutput(next_token_logits=logits)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for _ in weights:
            pass

    def _sample_next_token_on_device(self, generated: int) -> int:
        if generated >= self.max_generated_tokens:
            return self.eos_token_id

        first_regular_token_id = 4
        return first_regular_token_id + (
            generated % (self.vocab_size - first_regular_token_id)
        )


EntryClass = TenstorrentDeepSeekR10528ForCausalLM
