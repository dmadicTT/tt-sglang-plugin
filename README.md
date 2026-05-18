# Tenstorrent SGLang Plugin

Out-of-tree SGLang plugin that registers Tenstorrent as an SRT platform and
ships a CPU mock model for measuring SGLang's per-token serving overhead.

## Quick start

**1. Clone the repo.**

```bash
git clone https://github.com/dmadicTT/tt-sglang-plugin.git
cd tt-sglang-plugin
```

**2. Install.** Builds CPU-only SGLang from source and installs the plugin
into `./.venv`. Requires `uv`, `git`, and a C++ toolchain.

```bash
./install.sh
```

**3. Run the benchmark.** Starts the bundled mock, runs `vllm bench serve` at
single-user greedy decoding, and tears the server down on exit.

```bash
./bench.sh
```

**4. Read the output.** The mock's `forward()` sleeps for a known time —
`1000 / mock_tsu` milliseconds per token. Default `mock_tsu = 500`, so the
**expected TPOT/ITL is 2 ms / token**. Anything above that is SGLang's
per-token overhead:

```
overhead = reported Median TPOT − (1000 / TSU)
```

For example, if `./bench.sh` reports:

```
Median TPOT (ms):  2.45
```

then

```
overhead = 2.45 − 2.00 = 0.45 ms / token
```

That's SGLang's scheduler floor at single-user on this host.

## Variations

- **Tighter forward floor.** `TSU=1000 ./bench.sh` → 1 ms / token expected.
- **Client default sampling** (top-p / top-k / multinomial instead of greedy):
  `SAMPLING=default ./bench.sh`. The result should be the same, because the
  plugin's on-device sampling shim makes TPOT invariant to client sampling
  config.
- **Pass-through args to `vllm bench serve`:**
  `./bench.sh --random-output-len 1024 --num-prompts 8`.

## Going deeper

- Methodology, the on-device sampling shim, caveats, and limits of the
  measurement: [`HOW_WE_MEASURE_OVERHEAD.md`](HOW_WE_MEASURE_OVERHEAD.md).
- Plugin internals (activation contract, file map, mock env vars):
  [`sglang_tenstorrent/`](sglang_tenstorrent/) and its
  [README](sglang_tenstorrent/README.md).
