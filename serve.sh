#!/usr/bin/env bash
# Launch sglang serve with the bundled Tenstorrent mock model.
#
# Examples:
#     ./serve.sh
#     PORT=30001 ./serve.sh
#     SGLANG_TENSTORRENT_MOCK_TSU=1000 ./serve.sh        # 1 ms / token floor
#     SGLANG_TENSTORRENT_MOCK_LOG=debug ./serve.sh       # verbose mock logs
#     ./serve.sh --some-extra-flag                       # extra flags forward to sglang
#
# Environment overrides:
#     PORT, HOST, DEVICE, MAX_TOTAL_TOKENS, CONTEXT_LENGTH
#     SGLANG_TENSTORRENT_MOCK_TSU      - tokens / sec / user
#     SGLANG_TENSTORRENT_MOCK_LOG      - "info" or "debug" (anything truthy = debug)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOCK_MODEL="${REPO_ROOT}/sglang_tenstorrent/mock_model"

export SGLANG_TENSTORRENT_MOCK="${SGLANG_TENSTORRENT_MOCK:-1}"
export SGLANG_USE_CPU_ENGINE="${SGLANG_USE_CPU_ENGINE:-1}"
export SGLANG_PLATFORM="${SGLANG_PLATFORM:-tenstorrent}"

PORT="${PORT:-30050}"
HOST="${HOST:-127.0.0.1}"
DEVICE="${DEVICE:-cpu}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"

exec sglang serve \
    --model-path "${MOCK_MODEL}" \
    --served-model-name deepseek-ai/DeepSeek-R1-0528 \
    --tokenizer-path deepseek-ai/DeepSeek-R1-0528 \
    --load-format dummy \
    --device "${DEVICE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --context-length "${CONTEXT_LENGTH}" \
    "$@"
