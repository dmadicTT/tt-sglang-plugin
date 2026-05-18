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
