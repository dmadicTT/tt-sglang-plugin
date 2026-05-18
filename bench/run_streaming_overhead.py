"""Measure SGLang per-token streaming overhead against the Tenstorrent mock model.

The mock's `forward()` calls `time.sleep(1 / mock_tsu)` so the per-token floor
is known. `mock_tsu` is tokens per second per user; e.g. tsu=500 ⇒ 2 ms /
token. Anything the bench reports above that floor is SGLang overhead
(scheduler dispatch, IPC between tokenizer/scheduler/detokenizer, HTTP streaming,
etc).

Usage:
    .venv/bin/python bench/run_streaming_overhead.py
    .venv/bin/python bench/run_streaming_overhead.py --tsu 1000   # 1 ms / token floor
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_PACKAGE = REPO_ROOT / "sglang_tenstorrent"
MOCK_MODEL_DIR = PLUGIN_PACKAGE / "mock_model"
SERVED_MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528"
SERVER_READY_LINE = "The server is fired up and ready to roll"
MOCK_TSU_ENV = "SGLANG_TENSTORRENT_MOCK_TSU"


def pick_free_port() -> int:
    # sglang derives an internal gRPC port as serve_port + 10000, so the serve port
    # must be <= 55535. Probe random ports in [20000, 55000) until one binds.
    import random
    rng = random.Random()
    for _ in range(50):
        port = rng.randint(20000, 55000)
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("could not find a free port in [20000, 55000)")


def resolve_mock_tsu(cli_tsu: float | None) -> float:
    """Pick mock_tsu using the same precedence as the model: CLI > env > config.json."""
    if cli_tsu is not None:
        return cli_tsu
    env_value = os.environ.get(MOCK_TSU_ENV)
    if env_value:
        return float(env_value)
    config = json.loads((MOCK_MODEL_DIR / "config.json").read_text())
    return float(config["mock_tsu"])


def wait_until_ready(log_path: Path, server: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"sglang serve exited (code={server.returncode}) before signalling ready; "
                f"see {log_path}"
            )
        try:
            text = log_path.read_text(errors="ignore")
        except FileNotFoundError:
            text = ""
        if SERVER_READY_LINE in text:
            return
        time.sleep(0.5)
    raise TimeoutError(f"sglang serve did not signal ready within {timeout_s}s; see {log_path}")


def start_server(port: int, log_path: Path, max_total_tokens: int, context_length: int, mock_tsu: float) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(
        SGLANG_TENSTORRENT_MOCK="1",
        SGLANG_USE_CPU_ENGINE="1",
        SGLANG_PLATFORM="tenstorrent",
    )
    env[MOCK_TSU_ENV] = repr(mock_tsu)
    venv_bin = Path(sys.executable).parent
    sglang_cli = venv_bin / "sglang"
    cmd = [
        str(sglang_cli),
        "serve",
        "--model-path",
        str(MOCK_MODEL_DIR),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--load-format",
        "dummy",
        "--device",
        "cpu",
        "--tokenizer-path",
        SERVED_MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-total-tokens",
        str(max_total_tokens),
        "--context-length",
        str(context_length),
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]
    log_handle = log_path.open("wb")
    return subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )


def run_bench(port: int, num_prompts: int, output_len: int, concurrency: int) -> str:
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        SERVED_MODEL_NAME,
        "--tokenizer",
        SERVED_MODEL_NAME,
        "--dataset-name",
        "random-ids",
        "--random-input",
        "32",
        "--random-output",
        str(output_len),
        "--random-range-ratio",
        "1.0",
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"bench_serving exited {result.returncode}\n{out}")
    return out


_metric_pattern = re.compile(
    r"^(?P<label>Mean ITL|Median ITL|P95 ITL|P99 ITL|Max ITL|Mean TPOT|Median TPOT|P99 TPOT)\s*\(ms\):\s*(?P<value>[\d.]+)",
    re.MULTILINE,
)


def parse_metrics(bench_output: str) -> dict[str, float]:
    return {m.group("label"): float(m.group("value")) for m in _metric_pattern.finditer(bench_output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--output-len", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--server-log", type=Path, default=Path("/tmp/sglang-bench-server.log"))
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument(
        "--tsu",
        type=float,
        default=None,
        help=f"mock_tsu (tokens/sec/user). Overrides ${MOCK_TSU_ENV} and config.json.",
    )
    args = parser.parse_args()

    if args.output_len > args.context_length - 64:
        parser.error("output_len must fit inside context_length minus a small prompt")
    if args.tsu is not None and args.tsu <= 0:
        parser.error("--tsu must be > 0")

    port = args.port or pick_free_port()
    mock_tsu = resolve_mock_tsu(args.tsu)
    floor_ms = 1000.0 / mock_tsu

    print(f"[bench] mock_tsu={mock_tsu:.3f} tokens/sec/user (floor {floor_ms:.3f} ms / token)")
    print(f"[bench] starting sglang serve on 127.0.0.1:{port}, log -> {args.server_log}")

    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    if args.server_log.exists():
        args.server_log.unlink()
    server = start_server(
        port=port,
        log_path=args.server_log,
        max_total_tokens=args.max_total_tokens,
        context_length=args.context_length,
        mock_tsu=mock_tsu,
    )

    try:
        wait_until_ready(args.server_log, server, args.ready_timeout)
        print(f"[bench] server ready, running bench_serving: "
              f"num_prompts={args.num_prompts} output_len={args.output_len} concurrency={args.concurrency}")
        out = run_bench(
            port=port,
            num_prompts=args.num_prompts,
            output_len=args.output_len,
            concurrency=args.concurrency,
        )
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGINT)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=5)

    print("\n========== bench_serving output ==========")
    print(out)
    print("==========================================")

    metrics = parse_metrics(out)
    if not metrics:
        print("[bench] could not parse ITL metrics from bench_serving output")
        return 1

    median_itl = metrics.get("Median ITL", float("nan"))
    p99_itl = metrics.get("P99 ITL", float("nan"))
    mean_itl = metrics.get("Mean ITL", float("nan"))

    print(f"\n[overhead summary]")
    print(f"  mock floor                : {floor_ms:.3f} ms")
    print(f"  mean   ITL                : {mean_itl:.3f} ms   (overhead {mean_itl - floor_ms:+.3f} ms)")
    print(f"  median ITL                : {median_itl:.3f} ms   (overhead {median_itl - floor_ms:+.3f} ms)")
    print(f"  p99    ITL                : {p99_itl:.3f} ms   (overhead {p99_itl - floor_ms:+.3f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
