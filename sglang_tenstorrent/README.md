# SGLang Tenstorrent Plugin

Out-of-tree SGLang plugin that registers Tenstorrent as an SRT platform and ships a
Tenstorrent adapter for `deepseek-ai/DeepSeek-R1-0528`. A self-contained mock
model is bundled so the plugin can be exercised on any host without Tenstorrent
hardware -- useful for measuring SGLang's per-token overhead.

## Requirements

- `sglang >= 0.5.11` -- the release that introduced the `sglang.srt.platforms`
  and `sglang.srt.plugins` entry-point groups this plugin hooks into. Earlier
  releases (including `0.5.9`) do not import this plugin.
- For real Tenstorrent inference: a working `ttnn` install. With `ttnn` absent
  the plugin only activates when `SGLANG_TENSTORRENT_MOCK=1` is set.

## Install

The plugin is distributed as source only -- install it from a checkout of this
repository. SGLang itself comes from PyPI in the typical case (see below for
the CPU caveat).

The commands below use [`uv`](https://docs.astral.sh/uv/); plain `pip` works
identically (drop the `uv ` prefix). `uv pip` auto-discovers a `.venv` in the
current directory, so no manual activation is needed.

```bash
# 1. create a venv (skip if you already have one)
uv venv

# 2. install sglang into it (see "SGLang setup" below for the CPU caveat)
uv pip install --prerelease=allow 'sglang>=0.5.11'

# 3. install this plugin from a checkout
git clone <this-repo> tt-sglang-plugin
uv pip install -e ./tt-sglang-plugin
```

There's no global registration step. The plugin advertises two entry points
(`sglang.srt.platforms` and `sglang.srt.plugins`) which SGLang discovers
automatically when it starts.

### SGLang setup

**GPU host, or any host with `libcuda.so.1` available:** the standard SGLang
wheel works fine, including for the bundled mock model -- the mock just sleeps,
so the GPU is never actually exercised.

```bash
uv pip install --prerelease=allow 'sglang>=0.5.11'   # --prerelease=allow is currently required by sglang's flash-attn-4 dep
```

You can run the mock with `--device cuda` (it allocates a trivial logits tensor
on the GPU and otherwise does nothing) or `--device cpu`.

**CPU-only host (no `libcuda.so.1`):** the standard SGLang wheel cannot be
imported -- its `sgl_kernel` dependency loads CUDA libraries at module-import
time and refuses to fall through. There is no `sglang-cpu` wheel on PyPI; the
CPU build has to be compiled from source. Follow
[`docs/platforms/cpu_server.md`][cpu-docs] in the SGLang repository. Once
`import sglang` works in your environment the plugin install above is
unchanged.

[cpu-docs]: https://github.com/sgl-project/sglang/blob/main/docs/platforms/cpu_server.md

## How activation works

This package declares two entry points in `pyproject.toml`:

```toml
[project.entry-points."sglang.srt.platforms"]
tenstorrent = "sglang_tenstorrent:activate"

[project.entry-points."sglang.srt.plugins"]
tenstorrent = "sglang_tenstorrent:register"
```

At server startup SGLang discovers them automatically. `activate()` returns the
Tenstorrent platform class when either `ttnn` is importable or
`SGLANG_TENSTORRENT_MOCK=1` is set; otherwise it returns `None` and SGLang
ignores the plugin. `register()` adds the Tenstorrent model package to
`ModelRegistry` and wires the `served_model_name` startup hook.

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

## Real Tenstorrent inference

When `ttnn` is importable on the host, `activate()` returns the real platform
class and the plugin loads without any environment variable. The current model
adapter at `models/deepseek_r1_0528.py` is the mock; replace its `forward()`
with the real Tenstorrent execution path.

## Integration points

| File | Role |
| --- | --- |
| `__init__.py` | Thin entry point: `activate()`, `register()`, startup hooks. |
| `platform.py` | Hardware adapter: memory reporting, attention backend default, KV pool / paged allocator classes. |
| `models/deepseek_r1_0528.py` | Model adapter -- replace with real Tenstorrent execution. |
| `deepseek_r1_0528.py` | Helpers and paths for the bundled mock. |
| `mock_model/config.json` | Mock model config (vocab size, per-forward delay, etc.). |

## Notes

- **Device-side sampling.** SGLang currently expects `forward()` to return
  logits and samples on the host. The mock uses a one-hot compatibility shim.
  Production code should register a Tenstorrent sampler backend via
  `sglang.srt.layers.sampler.register_sampler_backend` and skip materialising
  full logits at the host boundary.
- **Mock vocab size.** Set to 131072 so the mock fits the DeepSeek-R1
  tokenizer (vocab 129,280). Adjust `mock_model/config.json` if you need to
  match a different tokenizer.
