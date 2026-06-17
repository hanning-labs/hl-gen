# code_switch — Development Plan (living checklist)

Fill the skeleton's stubs in dependency order so the pipeline produces, scores, and
persists a real `CSSample` end-to-end, then close the refinement loop and make it
testable. The orchestrator loop (`orchestrator.py::SynthesisPipeline.run`), data models,
config, agent ABCs, and the `LLMClient` protocol are already real — this work only
implements the bodies behind existing interfaces.

**Workflow:** one step at a time, human-in-the-loop. A step is not "done" until the user
has tested it and approved moving on. Update the checkboxes/status here as we go.

**LLM strategy:** local models first. Primary backend is an in-process `LocalClient`
using Hugging Face `transformers`, default model `Qwen/Qwen2.5-7B-Instruct`
(config-overridable to 14B). `ClaudeClient` stays a stub (API fallback for later). Agents
depend only on the `LLMClient` protocol, so the backend stays swappable.

Status legend: `[ ]` todo · `[~]` in progress (awaiting user test/approval) · `[x]` done & approved

---

## Priority 0 — Loop runs end-to-end and persists one sample (local model)

- [~] **P0.1 — `local` extra + `LocalClient`** (`llm/local.py`, `pyproject.toml`, `llm/__init__.py`)
  HF `transformers` backend satisfying the `LLMClient` protocol. Lazy-load model+tokenizer
  once; build prompt via `apply_chat_template`; run blocking `model.generate` inside
  `asyncio.to_thread` guarded by an `asyncio.Lock`; decode only new tokens; return
  `LLMResponse(text, model, usage, raw)`; forward `**kwargs` to `generate`. `ClaudeClient`
  stays a stub.
- [ ] **P0.2 — `FakeLLMClient` test double** (`tests/fakes.py`) — scripted client so agents are
  buildable/testable without loading Qwen.
- [ ] **P0.3 — Prompt-build + JSON-parse helper** (`prompting.py`) — render request/principles/
  tool_context/feedback into prompts; robustly extract+validate JSON (no native schema mode).
- [ ] **P0.4 — `GenerationAgent.generate`** (`agents/generation.py`).
- [ ] **P0.5 — Four scorer agents** (`agents/scorers.py`) — Fluency / Naturalness / CSRatio /
  SocialCulture; shared scoring helper + per-dimension rubrics returning `{score, rationale}`.
- [ ] **P0.6 — `SummarizeAgent.summarize`** (`agents/summarize.py`) — deterministic weighted mean
  + `passed` threshold; no model call.
- [ ] **P0.7 — Reference `SampleStore`** (`storage/file_store.py`) — JSONL/file store implementing
  the protocol; export from `storage/__init__.py`.
- [ ] **P0.8 — `AcceptanceAgent.accept`** (`agents/acceptance.py`) — attach scores+provenance,
  persist via store.

## Priority 1 — Close the refinement loop and make it testable

- [ ] **P1.1 — `RefinerAgent.refine`** (`agents/refiner.py`).
- [ ] **P1.2 — Per-agent unit tests with `FakeLLMClient`** (`tests/test_agents.py`).
- [ ] **P1.3 — Seed default `LinguisticPrinciples`** (`principles.py`).
- [ ] **P1.4 — Config + runnable example** (`examples/`, README) — `LocalClient` config surface
  (`model_id`, `device`, `dtype`, `max_new_tokens`); end-to-end demo.

## Priority 2 — Enrichment and hardening (deferrable)

- [ ] **P2.1 — Tool providers** (`tools/news.py`, `tools/social.py`).
- [ ] **P2.2 — Constrained JSON decoding** (grammar / logits processors).
- [ ] **P2.3 — Batch driver + corpus mapping** to `semantic_units`/`utterances`.
- [ ] **P2.4 — Robustness** — OOM/device fallbacks, parse-failure retries, batching, logging,
  and filling in `ClaudeClient.complete` as the API path.

---

## Verification

- Install: `cd code_switch && pip install -e ".[dev,local]"` (weights download lazily on first
  real `LocalClient` use).
- Existing loop tests stay green: `pytest` (`tests/test_pipeline_flow.py`).
- After P1.2: `pytest` runs `FakeLLMClient`-backed agent tests with no model download.
- After P0: live smoke — `build_default_pipeline(LocalClient(...), FileSampleStore(path))` →
  `await pipeline.run(request)` returns a `CSSample` (or `None`) and writes an accepted sample
  with scores+provenance. Use a smaller model / CPU on the dev box if 7B doesn't fit.
