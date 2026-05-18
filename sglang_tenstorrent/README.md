# SGLang Tenstorrent Plugin

Out-of-tree SGLang plugin that registers Tenstorrent as an SRT platform and
ships a mock model for measuring SGLang's per-token streaming overhead.

## Requirements

- `sglang >= 0.5.11` -- the release that introduced the `sglang.srt.platforms`
  and `sglang.srt.plugins` entry-point groups this plugin hooks into. Earlier
  releases (including `0.5.9`) do not import this plugin.
- `SGLANG_TENSTORRENT_MOCK=1` -- gates activation; the plugin is a no-op when
  unset. `serve.sh` sets it for you.

## Install

Tenstorrent hosts typically have no NVIDIA driver and so cannot import the
standard SGLang wheel (its `sgl_kernel` dlopens `libcuda.so.1` at module load).
SGLang does not publish a CPU wheel on PyPI, so both SGLang and its kernel
package must be built from source.

`install.sh` at the repo root automates the whole recipe. From a checkout:

```bash
./install.sh
source .venv/bin/activate
```

It creates `.venv/` (Python 3.12), clones SGLang at `v0.5.11` into `./sglang/`,
builds `sglang-cpu` and `sglang-kernel-cpu` (~40 s of C++ compile), and
installs this plugin editable. Re-runnable; reuses an existing venv and
sglang checkout if found. Requires `uv`, `git`, and a C++ toolchain. Override
defaults via env vars:

```bash
VENV_DIR=.venv-custom SGLANG_REPO=./vendor/sglang SGLANG_TAG=v0.5.11 ./install.sh
```

There's no global registration step. The plugin advertises two entry points
(`sglang.srt.platforms` and `sglang.srt.plugins`) which SGLang discovers
automatically when it starts. The script runs a final smoke check
(`activate()` should print the platform class).

## How activation works

This package declares two entry points in `pyproject.toml`:

```toml
[project.entry-points."sglang.srt.platforms"]
tenstorrent = "sglang_tenstorrent:activate"

[project.entry-points."sglang.srt.plugins"]
tenstorrent = "sglang_tenstorrent:register"
```

At server startup SGLang discovers them automatically. `activate()` returns
the Tenstorrent platform class when `SGLANG_TENSTORRENT_MOCK=1` is set;
otherwise it returns `None` and SGLang ignores the plugin. `register()` adds
the Tenstorrent model package to `ModelRegistry` and wires the
`served_model_name` startup hook.

There is no global configuration step -- if the package is installed in the
active environment, `sglang serve` picks it up.

## Run the bundled mock

The mock's `forward()` sleeps `1 / mock_tsu` seconds per call and returns a
synthetic token id. `mock_tsu` is **tokens per second per user** -- for example,
`mock_tsu=500` ⇒ 2 ms / token, `mock_tsu=1000` ⇒ 1 ms / token. The default
lives in `sglang_tenstorrent/mock_model/config.json` (currently `500.0`) and
can be overridden at runtime by setting `SGLANG_TENSTORRENT_MOCK_TSU`.

There's a `serve.sh` wrapper at the repo root that fills in the right flags:

```bash
./serve.sh                                       # CPU defaults, port 30050
PORT=30001 ./serve.sh                            # custom port
SGLANG_TENSTORRENT_MOCK_TSU=1000 ./serve.sh      # 1 ms / token floor
./serve.sh --tp-size 2                           # extra flags forward to sglang
```

The full `sglang serve` invocation it produces:

```bash
MOCK_MODEL=$(python -c "from sglang_tenstorrent.deepseek_r1_0528 import deepseek_r1_0528_mock_model_path; print(deepseek_r1_0528_mock_model_path())")

SGLANG_TENSTORRENT_MOCK=1 SGLANG_PLATFORM=tenstorrent \
SGLANG_TENSTORRENT_MOCK_TSU=500 \
  sglang serve \
    --model-path "$MOCK_MODEL" \
    --served-model-name deepseek-ai/DeepSeek-R1-0528 \
    --tokenizer-path deepseek-ai/DeepSeek-R1-0528 \
    --load-format dummy \
    --device cpu \
    --max-total-tokens 8192
```

`--max-total-tokens` is required: the mock's `hidden_size=1` makes per-token KV
trivially small, and without an explicit cap SGLang's automatic KV sizing tries
to allocate billions of slots and OOMs.

Verify with `GET /v1/models` -- you should see `deepseek-ai/DeepSeek-R1-0528`.

### Enable mock-model logs

Set `SGLANG_TENSTORRENT_MOCK_LOG` to see what the mock is doing -- one INFO
line per `__init__` plus one DEBUG line per `forward()` (batch size, request
keys, chosen token ids):

```bash
SGLANG_TENSTORRENT_MOCK_LOG=info  ./serve.sh    # init only
SGLANG_TENSTORRENT_MOCK_LOG=debug ./serve.sh    # init + every forward()
SGLANG_TENSTORRENT_MOCK_LOG=1     ./serve.sh    # alias for debug
```

The handler is installed on the mock model's logger with
`propagate=False`, so the extra lines don't go through SGLang's root logger
and can't trigger double-printing.

## Run the streaming-overhead benchmark

A `bench.sh` wrapper at the repo root forwards everything to
`bench/run_streaming_overhead.py`:

```bash
./bench.sh                                          # 4 prompts × 2048 tokens, concurrency 1
./bench.sh --tsu 1000                               # 1 ms / token floor
./bench.sh --concurrency 8 --num-prompts 16         # under load
./bench.sh --output-len 4096 --context-length 8192  # longer streams
./bench.sh --help                                   # full flag list
```

The script starts the mock, runs `sglang.bench_serving` against it, and prints
ITL against the mock's known per-token sleep floor. `--tsu` overrides
`mock_tsu` for a single run (the script propagates it to the server via
`SGLANG_TENSTORRENT_MOCK_TSU`). See the Python script's module docstring for
the full methodology and known limitations.

## Integration points

| File | Role |
| --- | --- |
| `__init__.py` | Thin entry point: `activate()`, `register()`, startup hooks. |
| `platform.py` | Hardware adapter: memory reporting, attention backend default, KV pool / paged allocator classes. |
| `models/deepseek_r1_0528.py` | Mock model adapter (sleep-based, returns synthetic token ids). |
| `deepseek_r1_0528.py` | Helpers and paths for the bundled mock. |
| `mock_model/config.json` | Mock model config (vocab size, per-forward delay, etc.). |

## Notes

- **Mock vocab size.** Set to 131072 so the mock fits the DeepSeek-R1
  tokenizer (vocab 129,280). Adjust `mock_model/config.json` if you need to
  match a different tokenizer.
