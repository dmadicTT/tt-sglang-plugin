# Measuring SGLang Scheduler Overhead

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
`/v1/chat/completions` endpoint, passing `--temperature 0` to force greedy
sampling (see the *Sampling* note below for why that is the right comparison
for DeepSeek). We read **TPOT** (time per output token, excluding the first
token) and compute:

```
overhead_per_token  =  median TPOT  −  1 / mock_tsu
```

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

## Measured floor: ~0.46 ms / token

`vllm bench serve` at concurrency = 1, ISL = 128, OSL = 2048, 4 prompts,
TSU = 500 (forward floor = 2.0 ms / token):

| Sampling                         | Median TPOT | Overhead vs floor |
| -------------------------------- | ----------: | ----------------: |
| `--temperature 0` (greedy)       |     2.46 ms |          +0.46 ms |
| default (server-side sampling)   |     4.79 ms |          +2.79 ms |

The greedy intercept (**~0.46 ms**) is SGLang's per-tick scheduler
bookkeeping floor. The default-sampling row pays an extra ~2.3 ms / token of
CPU-side sampler work; see the *Sampling* note below for why that doesn't
apply to real DeepSeek serving.

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

## Sampling: greedy vs default

`vllm bench serve` no longer sets `temperature=0` by default; it falls
through to whatever the server picks (top-p / top-k / multinomial over the
full 131 072-entry DeepSeek-R1 vocab). The ~2.3 ms gap between the two rows
above is **SGLang's CPU sampler cost on this host**, not anything in the
scheduler, the network, or the client.

**Note for DeepSeek:** since sampling for DeepSeek is done on the device,
it is only valid to compare the greedy sampling from the client perspective.
The ~2.3 ms gap for "default" sampling is a mock artifact — the mock returns
logits to SGLang, forcing the CPU sampler path that a real Tenstorrent
device would bypass regardless of the client's `temperature` setting.

## Running it

Start the server with the mock model:

```bash
SGLANG_TENSTORRENT_MOCK_TSU=500 ./serve.sh
```

In another shell, run the canonical greedy bench:

```bash
vllm bench serve \
    --model 'deepseek-ai/DeepSeek-R1-0528' \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 2048 \
    --num-prompts 4 \
    --max-concurrency 1 \
    --temperature 0 \
    --port 30050
```

Read **Median TPOT** from the output. Scheduler overhead =
`Median TPOT − 1000 / mock_tsu` (ms).

For the default-sampling reference, drop `--temperature 0` — but remember
that the extra ~2.3 ms it shows is a mock artifact (see *Sampling* above).
Always run at concurrency = 1 — see *Scope* above.
