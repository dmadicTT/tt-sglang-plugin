# Tenstorrent SGLang Plugin

Out-of-tree SGLang plugin that registers Tenstorrent as an SRT platform and
ships a CPU mock model for measuring SGLang's per-token serving overhead.

## Install

```bash
./install.sh
source .venv/bin/activate
```

[`install.sh`](install.sh) builds CPU-only SGLang from source and installs
this plugin editable into a fresh venv at `./.venv`. Idempotent — re-running
reuses an existing venv and sglang checkout. Requires `uv`, `git`, and a C++
toolchain. Override defaults via `VENV_DIR`, `SGLANG_REPO`, `SGLANG_TAG`.

## Serve the mock model

```bash
./serve.sh                                     # port 30050, mock_tsu=500
SGLANG_TENSTORRENT_MOCK_TSU=1000 ./serve.sh    # tighter per-token floor
```

The mock's `forward()` sleeps `1 / mock_tsu` seconds and returns synthetic
logits, giving SGLang a backend with a known, tunable forward cost.

## Benchmark SGLang's per-token overhead

```bash
./bench.sh                                     # canonical greedy bench
TSU=200 ./bench.sh                             # different forward floor
SAMPLING=default ./bench.sh                    # client default sampling
```

[`bench.sh`](bench.sh) starts the mock server, runs `vllm bench serve` at
single-user against `/v1/chat/completions`, and tears the server down on
exit. Subtract `1000 / TSU` ms from the reported Median TPOT to read off
SGLang's scheduler floor — currently **~0.45 ms / token**. See
[`bench/README.md`](bench/README.md) for the methodology and the on-device
sampling shim that makes the number invariant to client sampling config.

## Plugin code

[`sglang_tenstorrent/`](sglang_tenstorrent/) contains the plugin: SRT
platform adapter, mock model, and the `tenstorrent` sampling backend.
SGLang auto-discovers it through Python entry points when the venv has it
installed. See [`sglang_tenstorrent/README.md`](sglang_tenstorrent/README.md)
for the activation contract and a per-file map.
