# Measuring SGLang Scheduler Overhead

## Goal

Quantify the per-token wall-clock cost that SGLang's serving stack adds *on top
of* model compute — i.e. what a backend with arbitrarily fast forwards would
still pay per generated token. This bounds the throughput ceiling any future
Tenstorrent backend can reach through SGLang's CPU scheduler path.

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

We then drive the server with `sglang.bench_serving` and read **ITL**
(inter-token latency) and **TPOT** (time per output token, excl. first token).
The scheduler overhead at a given concurrency is:

```
overhead_per_token  =  measured_ITL  −  1 / mock_tsu
```

By sweeping `mock_tsu` and concurrency we separate three components:

| Knob                | What it isolates                                          |
| ------------------- | --------------------------------------------------------- |
| Vary `mock_tsu`     | Linearity of overhead vs. forward cost (intercept = floor)|
| Vary concurrency    | How much overhead amortises across requests per tick      |
| `--disable-overlap-schedule` | Whether overlap deferral helps on CPU (expected: no) |

## Why this works

- **Forward is a single tunable constant.** Sweep `mock_tsu`, fit the curve;
  the y-intercept is the bookkeeping floor.
- **Dummy weights + cached logits buffer** mean the model contributes zero CPU
  cycles besides the sleep. Anything above the floor is SGLang.
- **`process_batch_result_decode` runs against real CPU tensors**, so sampler
  work, KV updates, and IPC traffic are all measured for real — nothing is
  short-circuited.
- **CPU engine has no overlap** (`forward_stream` / `schedule_stream` are
  no-ops on CPU per `scheduler.py:1502-1503`), so the scheduler tick is
  exposed serially. The measurement is exactly what a CPU user pays.

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
4. **Constant forward time regardless of batch size.** A real model's forward
   grows with batch; the mock doesn't. Fine for per-step overhead at fixed
   concurrency, misleading for throughput-vs-batch extrapolation.
5. **Single-stream amplifies per-token overhead.** At concurrency 1, one
   scheduler tick produces one token. At concurrency 32, the same tick
   produces 32 tokens → per-token overhead drops ~32×. Always report the
   concurrency the number was measured at.

## Sampling defaults dominate the reported overhead

This script drives `/v1/chat/completions` via `sglang.bench_serving --backend
sglang-oai-chat`, which injects `temperature=0` into every request
(`bench_serving.py:404-405`). That short-circuits the sampler to argmax. Other
benchmark tools (`vllm bench`, `guidellm`) **do not** set temperature, so the
server falls through to the full sampling pipeline (top-p / top-k / multinomial)
over the full vocab (131 072 entries for DeepSeek-R1).

Same server, same workload (ISL=128, OSL=1000, concurrency=1, TSU=500),
single-stream curl in **non-streaming** mode (so no client/network involved):

| Sampling                       | Wall-clock for 1000 tokens | TPOT     | Overhead vs 2 ms floor |
| ------------------------------ | -------------------------: | -------: | ---------------------: |
| `temperature=0` (greedy)       |                  2446.3 ms |  2.45 ms |               +0.45 ms |
| server default (full sampling) |                  4675.9 ms |  4.68 ms |               +2.68 ms |

The ~2.2 ms gap is pure server-side sampler cost on CPU; it has nothing to do
with the client. That also explains the apparent gap between tools:

| Client                                | Sampling            | Median TPOT |
| ------------------------------------- | ------------------- | ----------: |
| `sglang.bench_serving` (this script)  | injects `temp=0`    |     2.45 ms |
| `vllm bench --backend openai-chat`    | server default      |     4.61 ms |
| `guidellm --backend openai_http`      | server default      |     4.70 ms |
| `curl` (greedy)                       | `temperature: 0`    |     2.45 ms |
| `curl` (default)                      | server default      |     4.68 ms |

So this script measures **SGLang's scheduler floor in isolation** (~0.5 ms)
by stripping out the sampler. For an end-user TPOT estimate on DeepSeek-R1
with realistic sampling, add the **~2.2 ms sampler cost** on top.

## Running it

The bench script handles server lifecycle, port allocation, and metric
parsing:

```bash
# default: tsu from config.json (500), concurrency 1, output_len 2048
.venv/bin/python bench/run_streaming_overhead.py

# tighter floor
.venv/bin/python bench/run_streaming_overhead.py --tsu 1000

# concurrency sweep
for c in 1 4 16 64; do
    .venv/bin/python bench/run_streaming_overhead.py --tsu 500 --concurrency $c
done
```

Output ends with an `[overhead summary]` block showing mean/median/p99 ITL
and their delta vs. the known floor.
