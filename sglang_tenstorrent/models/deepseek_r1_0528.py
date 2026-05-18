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

# Pre-tokenized Wikipedia-style passage the mock emits cyclically so every
# request streams the same recognisable content instead of a random walk through
# the vocab. Tokenized once with the deepseek-ai/DeepSeek-R1-0528 tokenizer (193
# tokens, max id 125517) so we don't need transformers / a tokenizer at model
# init time. To regenerate, run:
#
#     from transformers import AutoTokenizer
#     t = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528")
#     t.encode(ELEPHANT_TEXT, add_special_tokens=False)
#
# Source text -- paraphrase of the lead section of
# https://en.wikipedia.org/wiki/Elephant:
#
#     Elephants are the largest living land animals. Three living species are
#     currently recognised: the African bush elephant, the African forest
#     elephant, and the smaller Asian elephant. They are the only surviving
#     members of the family Elephantidae and the order Proboscidea; extinct
#     relatives include mammoths and mastodons. Distinctive features of
#     elephants include a long proboscis called a trunk, tusks, large ear
#     flaps, massive legs, and tough but sensitive grey skin. The trunk is
#     used for breathing, bringing food and water to the mouth, and grasping
#     objects. Tusks, derived from the incisor teeth, serve as weapons and as
#     tools for moving objects and digging. The large ear flaps help maintain
#     a constant body temperature as well as supporting communication. The
#     pillar-like legs carry their great weight.
#
# Note: the deepseek-r1 tokenizer wrapper in transformers drops whitespace on
# encode -> decode roundtrip, so the streamed text comes out word-concatenated.
# Content is still the elephant passage above.
ELEPHANT_TOKEN_IDS: tuple[int, ...] = (
    57689, 57524, 591, 1805, 26423, 416, 86228, 1831, 276, 18450, 16, 1636, 266, 317,
    340, 1045, 35369, 591, 125517, 63177, 2987, 28, 1805, 81063, 31757, 263, 302, 26750,
    55295, 81063, 1486, 3700, 302, 26750, 40280, 1805, 26758, 264, 81960, 13768, 26750,
    103979, 591, 1805, 16132, 27797, 88, 2331, 74531, 39499, 263, 33811, 57689, 26750,
    13354, 458, 1805, 4010, 45411, 9948, 28860, 29, 3405, 6149, 4419, 6261, 261, 1104,
    381, 4422, 48729, 458, 79, 648, 401, 1054, 5249, 435, 6149, 505, 69437, 2154, 13768,
    57524, 5211, 60114, 39634, 9948, 3487, 10546, 268, 84, 7048, 20197, 349, 813, 14,
    40372, 707, 1668, 3471, 28558, 624, 505, 3743, 85, 40280, 86, 1446, 5887, 85, 28118,
    28411, 922, 8705, 14170, 3050, 7048, 278, 6497, 2251, 3836, 50189, 16181, 981, 288,
    49295, 458, 9372, 60737, 2868, 2960, 40280, 73, 5171, 22153, 42891, 5903, 349, 813,
    14, 1514, 2419, 5356, 1805, 2769, 278, 20612, 1089, 14, 82136, 306, 971, 68970, 458,
    648, 13397, 794, 23049, 42891, 458, 24222, 5426, 14170, 40372, 707, 1668, 3471,
    4247, 7787, 499, 2260, 19092, 41931, 7193, 88634, 306, 8807, 624, 27232, 288, 52511,
    14170, 49241, 287, 11727, 3743, 1822, 19879, 39283, 52574, 11026, 16,
)

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
            "vocab_size=%d, max_generated_tokens=%d, fixed_text_tokens=%d",
            self.mock_tsu,
            self.token_delay_seconds,
            self.vocab_size,
            self.max_generated_tokens,
            len(ELEPHANT_TOKEN_IDS),
        )

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
        batch_size = len(req_pool_indices)
        device = forward_batch.seq_lens.device

        self._ensure_logits_buffer(batch_size, device)
        # Reset cells we wrote on the previous call (no-op on a fresh buffer).
        for row, tok in self._last_writes:
            self._logits_buffer[row, tok] = -1.0e9
        self._last_writes.clear()

        chosen: list[int] = []
        for row, request_key in enumerate(request_keys):
            generated = self._generated_by_request.get(request_key, 0)
            token_id = self._sample_next_token_on_device(generated)
            if token_id != self.eos_token_id:
                self._generated_by_request[request_key] = generated + 1
            # Compatibility shim: SGLang currently samples from logits, while
            # Tenstorrent will already have sampled token IDs on device.
            self._logits_buffer[row, token_id] = 0.0
            self._last_writes.append((row, token_id))
            chosen.append(token_id)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "forward: batch=%d keys=%s tokens=%s",
                batch_size,
                request_keys if batch_size <= 4 else f"{request_keys[:4]}...",
                chosen if batch_size <= 4 else f"{chosen[:4]}...",
            )

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
        return ELEPHANT_TOKEN_IDS[generated % len(ELEPHANT_TOKEN_IDS)]


EntryClass = TenstorrentDeepSeekR10528ForCausalLM
