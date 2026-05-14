# SGLang Tenstorrent Plugin

This package is an out-of-tree SGLang plugin scaffold for Tenstorrent hardware.
It currently includes a local mock that is served as `deepseek-ai/DeepSeek-R1-0528`.

## Install

From the SGLang repository root:

```bash
pip install -e plugins/tenstorrent
```

## Run The DeepSeek R1 Mock

The mock path is bundled inside this package:

```bash
MOCK_MODEL=$(python3 -c "from sglang_tenstorrent.deepseek_r1_0528 import deepseek_r1_0528_mock_model_path; print(deepseek_r1_0528_mock_model_path())")
SGLANG_TENSTORRENT_MOCK=1 SGLANG_PLATFORM=tenstorrent \
  sglang serve \
  --model-path "$MOCK_MODEL" \
  --served-model-name deepseek-ai/DeepSeek-R1-0528 \
  --load-format dummy \
  --device cpu \
  --skip-tokenizer-init
```

The mock emits one token every 20 ms and stops after 16 generated tokens.

## Integration Points

- `__init__.py` is the plugin entry point. Keep it thin: activation, model registration, and startup hooks.
- `platform.py` is the hardware adapter. Add Tenstorrent device setup, memory reporting, backend defaults, graph support, and worker initialization here.
- `models/deepseek_r1_0528.py` is the model adapter. Replace the mock forward path with real Tenstorrent execution for DeepSeek R1 0528.
- `deepseek_r1_0528.py` contains helper constants and paths for the bundled DeepSeek R1 0528 mock.

## Device-Side Sampling

SGLang currently expects model `forward()` to return logits and then calls its sampler.
For Tenstorrent device-side sampling, the current mock uses a compatibility shim:
the model selects the token ID first, then returns one-hot logits so SGLang's sampler
chooses the same token.

For a production integration, prefer adding a Tenstorrent sampler backend with
`sglang.srt.layers.sampler.register_sampler_backend(...)` and pass device-sampled
token IDs across the model-to-sampler boundary without materializing full logits.

## Useful Validation

Run `/v1/models` after startup and verify the served model ID is:

```text
deepseek-ai/DeepSeek-R1-0528
```

For performance debugging, collect per-token timestamps on the client and add timing
around Tenstorrent queue submit, device execution, synchronization, and token transfer.
