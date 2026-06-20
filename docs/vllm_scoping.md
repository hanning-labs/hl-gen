# Scoping: vLLM as Core Backend

## What vLLM Brings

**PagedAttention** — KV cache is stored in non-contiguous "pages" rather than a single preallocated tensor. Eliminates memory waste from padding (our current left-padding approach wastes GPU memory proportional to the longest sequence in the batch) and from fragmentation. Practical effect: you can run larger effective batch sizes in the same VRAM.

**Continuous batching** — Incoming requests slot into the active batch as soon as a sequence finishes, rather than waiting for the entire current batch to drain. Our `_drain_loop` fires on a fixed `batch_timeout_sec` window; vLLM's scheduler is adaptive and has no such latency floor.

**True async engine** (`AsyncLLMEngine`) — Natively async; no `asyncio.to_thread` wrapping required. Each `engine.generate()` call is a coroutine that can be awaited directly, removing the need for the `_BatchItem` queue entirely.

**Built-in tensor parallelism** — `tensor_parallel_size=N` shards the model across N GPUs. Currently we're bottlenecked to one GPU; this is the only path to multi-GPU without custom sharding code.

**Native guided decoding** — vLLM integrates `outlines` for JSON schema–constrained generation. Cleaner API than `lm-format-enforcer`'s `prefix_allowed_tokens_fn` at the batch level (no need to build `combined_prefix_fn` routing by `batch_id`).

---

## Full Scope of Changes

### New file: `llm/vllm_client.py`
- `VLLMClient` implementing the `LLMClient` protocol
- Lazy-imports `vllm` (behind optional dep)
- Initialises `AsyncLLMEngine` once on first call (analogous to `_ensure_loaded`)
- Extracts `json_schema` from kwargs; maps to vLLM's `GuidedDecodingParams`
- Awaits `engine.generate()` directly — no queue, no drain loop, no `to_thread`
- Normalises to `LLMResponse(text, model, usage, raw)`

### `batch.py`
- Add `VLLMClientConfig(BaseModel)` with fields: `model`, `dtype`, `max_new_tokens`, `tensor_parallel_size`, `max_num_seqs`, `gpu_memory_utilization`, `quantization`
- Remove or gate `LocalClientConfig`-specific fields (`max_batch_size`, `batch_timeout_sec`, `compile_model`) — they're meaningless for vLLM
- `BatchConfig.client` becomes a discriminated union (`Annotated[LocalClientConfig | VLLMClientConfig, Field(discriminator="backend")]`) — or simpler: two separate config keys (`local_client` / `vllm_client`) with only one active
- `examples/run_batch.py` inspects config type to instantiate the right client

### `pyproject.toml`
- New `vllm` optional dep group: `vllm>=0.6`
- Note: vLLM pins its own torch version — this will likely conflict with the current `torch==2.12.1+cu130` install and may require a separate venv or Docker image

### What goes away
- `_BatchItem` dataclass and `_drain_loop` in `LocalClient` are no longer needed for the vLLM path (though `LocalClient` stays for local dev without vLLM)
- `compile_model` flag (vLLM has its own compilation via `enforce_eager=False`)
- `lm-format-enforcer` as a hard dep for the vLLM path (replaced by `outlines`)

### What stays unchanged
- `LLMClient` protocol — the whole benefit of having a protocol
- All agents and `_complete_with_retry` — zero changes
- `orchestrator.py` — zero changes
- `storage/`, `models.py`, `config.py` — zero changes

---

## Tradeoffs

### Why you should integrate vLLM

| Factor | Impact |
|---|---|
| **OOM risk** | PagedAttention essentially eliminates CUDA OOM from batch sizing — our current `_run_batch` has a manual OOM retry that becomes unnecessary |
| **Multi-GPU** | `tensor_parallel_size=2` on two 3090s would roughly double throughput with no code changes beyond the config |
| **Scale** | At n=1000+, continuous batching vs our drain-loop is a 2–4× throughput difference (measured in the vLLM paper vs HuggingFace generate) |
| **Guided decoding** | outlines-based JSON constraint is per-request, not per-batch — cleaner than our `combined_prefix_fn` hack |
| **Architectural cleanliness** | The `_drain_loop` / `_BatchItem` complexity disappears; `VLLMClient.complete()` becomes ~20 lines |

### Why you shouldn't (yet)

| Factor | Impact |
|---|---|
| **Torch version conflict** | vLLM pins its own torch. Our venv runs `2.12.1+cu130`; vLLM releases lag. Likely requires a new venv or Docker, which is non-trivial operational overhead |
| **Our CUDA version is too new** | We just spent significant time failing to install flash-attn on cu130. vLLM has similar CUDA version constraints. cu130 support in vLLM likely isn't there yet |
| **Current scale doesn't need it** | At n=20 with 4 concurrent pipelines, our drain loop is not the bottleneck. The profiling data from torch.compile will tell us what actually is |
| **Continuous batching gain is marginal at low concurrency** | Continuous batching shines when you have hundreds of concurrent requests. At 4–8, it barely beats a fixed batch window |
| **More moving parts** | A separate `AsyncLLMEngine` in-process competes with our torch.compile'd model for CUDA context and memory management |
| **lm-format-enforcer is already working** | Rewiring guided decoding to outlines introduces risk for minimal benefit right now |

---

## Honest Recommendation

**Don't integrate vLLM now.** The `LLMClient` protocol means it's a clean swap when the time is right, but the CUDA 13.0 version compatibility is the same wall we hit with flash-attn — and vLLM's torch dependency is even harder to work around.

**The right trigger points for vLLM:**
1. You want to run on 2+ GPUs (`tensor_parallel_size`)
2. You're targeting n > 500 per run and wall time is the primary constraint
3. The profiling output shows `request_sec.mean` is still high after torch.compile

**The better near-term lever** is quantization (AWQ or GPTQ 4-bit). A quantized 7B model fits in ~4 GB VRAM instead of ~14 GB in bfloat16, which directly doubles the headroom for `max_batch_size` on a 3090 (24 GB). That's a config-level change with an existing HuggingFace model variant, no new infrastructure.

---

## If you do proceed

The cleanest integration path:
1. New venv (or Docker) with `vllm` pre-installed — don't mix with the existing venv
2. Add `llm/vllm_client.py` + `VLLMClientConfig` in `batch.py`
3. Discriminated union on `BatchConfig.client` so both backends are config-selectable
4. Run the same profiling script against both and compare `docs/profile_*.json` outputs
