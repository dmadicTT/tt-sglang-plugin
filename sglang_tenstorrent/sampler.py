"""Greedy-only sampler backend for the Tenstorrent plugin.

Tenstorrent hardware samples on-device and surfaces the chosen token id to
the host as a one-hot logits row (a compatibility shim against SGLang's
logits-in / ids-out Sampler interface). The host therefore only needs to
recover that single index — running the full softmax + sort + multinomial
pipeline over a 131k-entry vocab costs ~2 ms on CPU for no gain.

Enable via ``--sampling-backend tenstorrent``.
"""

from __future__ import annotations

from typing import List

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler, register_sampler_backend
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

TENSTORRENT_SAMPLING_BACKEND = "tenstorrent"


class TenstorrentSampler(Sampler):
    """Always returns ``argmax(logits)``, ignoring temperature / top-k / top-p.

    Real sampling parameters travel to the device as a side payload; the
    device samples and writes a one-hot logits row that argmax recovers in
    ~50 µs.
    """

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        del positions

        logits = logits_output.next_token_logits
        logits = self._preprocess_logits(logits, sampling_info)

        batch_next_token_ids = torch.argmax(logits, dim=-1)

        if return_logprob:
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            self._attach_logprobs_to_output(
                logits_output,
                logprobs,
                top_logprobs_nums,
                token_ids_logprobs,
                sampling_info,
                batch_next_token_ids,
            )

        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)
        return batch_next_token_ids


def register() -> None:
    register_sampler_backend(TENSTORRENT_SAMPLING_BACKEND, TenstorrentSampler)
