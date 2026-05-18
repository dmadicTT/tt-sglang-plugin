#!/usr/bin/env bash
# Streaming-overhead benchmark wrapper around bench/run_streaming_overhead.py.
#
# Examples:
#     ./bench.sh                                          # defaults from the Python script
#     ./bench.sh --tsu 1000                               # 1 ms / token floor
#     ./bench.sh --concurrency 8 --num-prompts 16         # under load
#     ./bench.sh --output-len 4096 --context-length 8192  # longer streams
#     SGLANG_TENSTORRENT_MOCK_LOG=info ./bench.sh         # also log mock init
#
# All args pass through to bench/run_streaming_overhead.py (run `./bench.sh --help`
# for the full list).
#
# Picks the Python interpreter in this order:
#     1. $PYTHON if set
#     2. ./.venv/bin/python if it exists
#     3. python3 / python on PATH
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/bench/run_streaming_overhead.py" "$@"
