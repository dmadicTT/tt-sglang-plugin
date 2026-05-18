# How We Measure SGLang Scheduler Overhead

## Goal

Quantify the **single-user** per-token wall-clock cost that SGLang's serving
stack adds *on top of* model compute — i.e. what a single in-flight request to
a backend with arbitrarily fast forwards would still pay per generated token.
This is the floor of TPOT a single user can ever observe through the SGLang
CPU path.

## Scope: single user only

This setup is **not** suitable for multi-user / batched throughput analysis.
The mock's `forward()` sleeps for a fixed `1 / mock_tsu` regardless of batch
size, so a batch of N concurrent requests still costs one sleep and produces
N tokens — implying that per-token wall-clock drops as `1 / N`. A real model's
forward time grows with batch size, so the mock's concurrent-user numbers
would be wildly optimistic. All measurements below are at concurrency = 1.

## Method

We isolate scheduler cost by giving SGLang a model whose forward time is a
known, tunable constant.

The mock model (`sglang_tenstorrent/models/deepseek_r1_0528.py`) implements
`forward()` as:

```python
time.sleep(1.0 / mock_tsu)   # mock_tsu = tokens / sec / user
# fill a cached logits buffer, return it
```

- `mock_tsu = 500` → forward floor = **2.0 ms / token**
- `mock_tsu = 1000` → forward floor = **1.0 ms / token**
- etc.

Everything else in SGLang runs unchanged: scheduler subprocess, tokenizer
manager, detokenizer subprocess, ZMQ IPC, sampler, KV pool bookkeeping, HTTP
streaming.

We drive the server with **`vllm bench serve`** against the OpenAI-compatible
`/v1/chat/completions` endpoint. The plugin's `TenstorrentSampler` makes the
result invariant to the client's `temperature` / `top_p` / `top_k` (see
*Sampling* below), so the default invocation passes `--temperature 0` purely
for clarity. We read **TPOT** (time per output token, excluding the first
token) and compute:

```
overhead_per_token  =  median TPOT  −  1 / mock_tsu
```

## Why this works

- **Forward is a single tunable constant.** Sweep `mock_tsu`, fit the curve;
  the y-intercept is the bookkeeping floor.
- **Dummy weights + cached logits buffer** mean the model contributes zero CPU
  cycles besides the sleep. Anything above the floor is SGLang.
- **`process_batch_result_decode` runs against real CPU tensors**, so the
  argmax sampler, KV updates, and IPC traffic are all measured for real.
  Heavyweight CPU sampling (softmax / top-p / top-k / multinomial) is
  short-circuited on purpose by `TenstorrentSampler` — see *Sampling* below.
- **CPU engine has no overlap** (`forward_stream` / `schedule_stream` are
  no-ops on CPU per `scheduler.py:1502-1503`), so the scheduler tick is
  exposed serially. The measurement is exactly what a CPU user pays.

## Measured floor: ~0.45 ms / token

`vllm bench serve` at concurrency = 1, ISL = 128, OSL = 2048, 4 prompts,
TSU = 500 (forward floor = 2.0 ms / token):

| Client config              | Median TPOT | Overhead vs floor |
| -------------------------- | ----------: | ----------------: |
| `--temperature 0` (greedy) |     2.45 ms |          +0.45 ms |
| default (no temperature)   |     2.46 ms |          +0.46 ms |

Both rows collapse to **~0.45 ms** because the plugin installs
`TenstorrentSampler` (`sglang_tenstorrent/sampler.py`), which always returns
`argmax(logits)` and ignores the client's `temperature` / `top_p` / `top_k`.
That mirrors real Tenstorrent serving — the device samples on-chip, so
SGLang's CPU sampler is bypassed regardless of what the client requests.
Without that shim, the default row would pay ~2.3 ms / token of CPU sampler
work (see *Sampling* below).

## Caveats

1. **`time.sleep()` granularity is ~1 ms on Linux.** Don't trust the floor
   above `mock_tsu ≈ 1000` (sub-ms sleeps become jittery). For tighter
   measurements, switch to a busy-wait inside `forward()`.
2. **`time.sleep()` releases the GIL.** Today's scheduler loop is
   single-threaded so this doesn't hide work, but it would if background
   threads were added.
3. **CPU-engine overhead ≠ GPU-engine overhead.** On GPU most of this work
   overlaps with kernel execution and is invisible. The number here is the
   floor that becomes exposed *if* device forwards are faster than the
   scheduler tick — useful for understanding when scheduler cost will
   bottleneck a fast backend.
4. **Single-user only.** See the *Scope* section above. Do not extrapolate
   the reported number to concurrent-user TPOT; the mock can't model
   batch-dependent forward time.

## Sampling: handled on-device

DeepSeek-R1 on a Tenstorrent device samples on-chip and writes a one-hot
logits row back to the host; SGLang's CPU sampler would be redundant. The
plugin models this with `TenstorrentSampler`, registered automatically
whenever the mock model is loaded (`sglang_tenstorrent/__init__.py` forces
`server_args.sampling_backend = "tenstorrent"`). The sampler short-circuits
to argmax and never touches softmax / top-p / top-k / multinomial.

Confirm it is active in the server log:

```
sampling_backend='tenstorrent'
```

If you ever benched this stack **without** the shim (e.g. against an older
plugin build, or with `--sampling-backend pytorch`), the default-sampling
client row would balloon by ~2.3 ms / token from CPU sampler work over the
131 072-entry vocab. That extra cost is a mock artifact: a real Tenstorrent
device would still bypass it.

## Running it

The repo ships a `bench.sh` wrapper that manages the server lifecycle and
runs `vllm bench serve` against the bundled mock:

```bash
# canonical: TSU=500, greedy, concurrency=1
./bench.sh

# tighter forward floor
TSU=1000 ./bench.sh

# default-sampling reference (no --temperature 0)
SAMPLING=default ./bench.sh

# pass-through args go to vllm bench serve
./bench.sh --random-output-len 1024 --num-prompts 8
```

Read **Median TPOT** from the output. Scheduler overhead =
`Median TPOT − 1000 / TSU` (ms). At TSU=500 expect ~2.45 ms TPOT → ~0.45 ms
overhead. Always at concurrency = 1 by design — see *Scope* above.
