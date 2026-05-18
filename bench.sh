#!/usr/bin/env bash
# Run `vllm bench serve` against the bundled mock SGLang server.
#
# Manages the full server lifecycle: starts ./serve.sh in the background,
# waits for it to become ready, runs the bench, and tears the server down
# on exit. Defaults to the canonical single-user greedy-decoding setup
# (see bench/README.md).
#
# Env overrides:
#     TSU=200 ./bench.sh              # tighter forward floor
#     SAMPLING=default ./bench.sh     # drop --temperature 0
#     PORT=20001 ./bench.sh
#     VLLM=/path/to/vllm ./bench.sh
#
# Extra positional args pass through to `vllm bench serve`:
#     ./bench.sh --random-output-len 1024
#     ./bench.sh --num-prompts 8
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TSU="${TSU:-500}"
PORT="${PORT:-30050}"
SAMPLING="${SAMPLING:-greedy}"
SERVER_LOG="${SERVER_LOG:-/tmp/sglang-bench-server.log}"
READY_TIMEOUT="${READY_TIMEOUT:-180}"

if [[ -z "${VLLM:-}" ]]; then
    if [[ -x "${REPO_ROOT}/.venv/bin/vllm" ]]; then
        VLLM="${REPO_ROOT}/.venv/bin/vllm"
    elif command -v vllm >/dev/null 2>&1; then
        VLLM="$(command -v vllm)"
    else
        echo "ERROR: vllm not found. Set VLLM=/path/to/vllm." >&2
        exit 1
    fi
fi

rm -f "${SERVER_LOG}"
SGLANG_TENSTORRENT_MOCK_TSU="${TSU}" PORT="${PORT}" \
    "${REPO_ROOT}/serve.sh" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    pkill -INT -f "sglang.*serve.*mock_model" 2>/dev/null || true
    for _ in {1..15}; do
        ss -ltn 2>/dev/null | grep -q ":${PORT}" || break
        sleep 1
    done
}
trap cleanup EXIT

DEADLINE=$(( $(date +%s) + READY_TIMEOUT ))
until grep -q "Uvicorn running on http://127.0.0.1:${PORT}" "${SERVER_LOG}" 2>/dev/null; do
    if (( $(date +%s) > DEADLINE )); then
        echo "ERROR: server didn't become ready within ${READY_TIMEOUT}s." >&2
        tail -20 "${SERVER_LOG}" >&2
        exit 1
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: server died before becoming ready." >&2
        tail -20 "${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 2
done

SAMPLING_ARGS=()
case "${SAMPLING}" in
    greedy)  SAMPLING_ARGS+=(--temperature 0) ;;
    default) : ;;
    *) echo "ERROR: SAMPLING must be 'greedy' or 'default', got '${SAMPLING}'" >&2; exit 1 ;;
esac

echo "[bench] TSU=${TSU} sampling=${SAMPLING} port=${PORT}"

"${VLLM}" bench serve \
    --model 'deepseek-ai/DeepSeek-R1-0528' \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 2048 \
    --num-prompts 4 \
    --max-concurrency 1 \
    --port "${PORT}" \
    "${SAMPLING_ARGS[@]}" \
    "$@"
