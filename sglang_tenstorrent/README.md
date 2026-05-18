# sglang_tenstorrent

SGLang plugin package: registers Tenstorrent as an SRT platform, ships the
mock model, and installs the `tenstorrent` sampling backend.

For install / serve / bench instructions, see the [repo-root
README](../README.md).

## Activation

Two entry points (declared in `pyproject.toml`):

```toml
[project.entry-points."sglang.srt.platforms"]
tenstorrent = "sglang_tenstorrent:activate"

[project.entry-points."sglang.srt.plugins"]
tenstorrent = "sglang_tenstorrent:register"
```

SGLang discovers these at startup. Activation is gated by
`SGLANG_TENSTORRENT_MOCK=1` — otherwise `activate()` returns `None` and the
plugin is a no-op. `serve.sh` sets it.

`register()` does three things:

1. Adds `sglang_tenstorrent.models` to `ModelRegistry`.
2. Registers `TenstorrentSampler` as the `tenstorrent` sampling backend.
3. Installs a startup hook (`_apply_mock_model_identity`) that pins
   `served_model_name = "deepseek-ai/DeepSeek-R1-0528"` and re-applies
   `sampling_backend = "tenstorrent"` after `ServerArgs._handle_cpu_backends`
   clobbers it back to `pytorch`.

## File map

| File | Role |
| --- | --- |
| `__init__.py` | Entry points, activation gate, startup hook. |
| `platform.py` | Platform adapter: memory, attention backend default, KV / paged allocator. |
| `models/deepseek_r1_0528.py` | Mock model: `forward()` sleeps `1 / mock_tsu` s, returns synthetic logits. |
| `deepseek_r1_0528.py` | Helpers for identifying the bundled mock by path. |
| `sampler.py` | `TenstorrentSampler` — always returns `argmax(logits)`, modeling on-device sampling. |
| `mock_model/config.json` | Mock config (vocab=131072 to fit DeepSeek-R1, `mock_tsu`, max generated tokens). |

## Mock knobs

| Env var | Default | Effect |
| --- | --- | --- |
| `SGLANG_TENSTORRENT_MOCK` | unset | Must be `1` for the plugin to activate. |
| `SGLANG_TENSTORRENT_MOCK_TSU` | 500 (from `mock_model/config.json`) | Tokens/sec/user. Mock sleeps `1 / TSU` per forward. |
| `SGLANG_TENSTORRENT_MOCK_LOG` | unset | `info` or `debug` enables mock-model log lines. |

`--max-total-tokens` must be set when serving: the mock's `hidden_size=1`
makes per-token KV trivially small, so without a cap SGLang's automatic
sizing tries to allocate billions of slots and OOMs. `serve.sh` sets it.
