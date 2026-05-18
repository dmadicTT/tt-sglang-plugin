"""DeepSeek R1 0528 mock model for Tenstorrent plugin bring-up."""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

MOCK_TSU_ENV = "SGLANG_TENSTORRENT_MOCK_TSU"
MOCK_LOG_ENV = "SGLANG_TENSTORRENT_MOCK_LOG"

logger = logging.getLogger(__name__)


def _configure_logger() -> None:
    raw = os.environ.get(MOCK_LOG_ENV, "").strip().lower()
    if not raw or raw in ("0", "false", "no"):
        return
    level = logging.INFO if raw == "info" else logging.DEBUG
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False


_configure_logger()


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
        # Logits buffer cached across forward() calls; allocated lazily so we
        # see the actual device. _last_writes tracks which (row, token_id) cells
        # we set to 0.0 last call so we can reset them back to -1e9 cheaply.
        self._logits_buffer: torch.Tensor | None = None
        self._last_writes: list[tuple[int, int]] = []
        logger.info(
            "mock model init: tsu=%.3f tokens/sec/user, delay=%.6f s, "
            "vocab_size=%d, max_generated_tokens=%d",
            self.mock_tsu,
            self.token_delay_seconds,
            self.vocab_size,
            self.max_generated_tokens,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        del input_ids, positions

        # Busy-wait, not time.sleep: on Linux time.sleep overshoots by a
        # constant ~50 µs that would land in the measured per-token overhead.
        # Busy-wait pegs a core but lands on the deadline to within ~0.1 µs.
        deadline = time.perf_counter() + self.token_delay_seconds
        while time.perf_counter() < deadline:
            pass

        req_pool_indices = forward_batch.req_pool_indices.detach().cpu().tolist()
        request_keys = forward_batch.rids or req_pool_indices
        batch_size = len(req_pool_indices)
        device = forward_batch.seq_lens.device

        self._ensure_logits_buffer(batch_size, device)
        # Reset cells we wrote on the previous call (no-op on a fresh buffer).
        for row, tok in self._last_writes:
            self._logits_buffer[row, tok] = -1.0e9
        self._last_writes.clear()

        for row, request_key in enumerate(request_keys):
            generated = self._generated_by_request.get(request_key, 0)
            token_id = self._sample_next_token_on_device(generated)
            if token_id != self.eos_token_id:
                self._generated_by_request[request_key] = generated + 1
            # Compatibility shim: SGLang currently samples from logits, while
            # Tenstorrent will already have sampled token IDs on device.
            self._logits_buffer[row, token_id] = 0.0
            self._last_writes.append((row, token_id))

        return LogitsProcessorOutput(
            next_token_logits=self._logits_buffer[:batch_size]
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for _ in weights:
            pass

    def _ensure_logits_buffer(self, batch_size: int, device: torch.device) -> None:
        if (
            self._logits_buffer is None
            or self._logits_buffer.shape[0] < batch_size
            or self._logits_buffer.device != device
        ):
            capacity = max(batch_size, 64)
            self._logits_buffer = torch.full(
                (capacity, self.vocab_size),
                -1.0e9,
                dtype=torch.float32,
                device=device,
            )
            # Fresh buffer -- any tracked writes pointed at the old one.
            self._last_writes.clear()
            logger.debug(
                "allocated logits buffer: shape=%s device=%s",
                tuple(self._logits_buffer.shape),
                device,
            )

    def _sample_next_token_on_device(self, generated: int) -> int:
        if generated >= self.max_generated_tokens:
            return self.eos_token_id

        first_regular_token_id = 4
        return first_regular_token_id + (
            generated % (self.vocab_size - first_regular_token_id)
        )


EntryClass = TenstorrentDeepSeekR10528ForCausalLM
