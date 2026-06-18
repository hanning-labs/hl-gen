# code_switch — Development Plan (living checklist)

Fill the skeleton's stubs in dependency order so the pipeline produces, scores, and
persists a real `CSSample` end-to-end, then close the refinement loop. The orchestrator
loop (`orchestrator.py::SynthesisPipeline.run`), data models, config, agent ABCs, and the
`LLMClient` protocol are already real — this work only implements the bodies behind
existing interfaces.

**Workflow:** one step at a time, human-in-the-loop. A step is not "done" until the user
has tested it and approved moving on. Update the checkboxes/status here as we go.

**LLM strategy:** local models first. Primary backend is an in-process `LocalClient`
using Hugging Face `transformers`, default model `Qwen/Qwen2.5-7B-Instruct`
(config-overridable to 14B). `ClaudeClient` stays a stub (API fallback for later). Agents
depend only on the `LLMClient` protocol, so the backend stays swappable.

**Testing approach:** validation is done against the real local model — the existing
orchestrator loop tests (`tests/test_pipeline_flow.py`, mock agents) must stay green, and
each new piece is exercised by the user via live runs. No mock/fake LLM doubles.

Status legend: `[ ]` todo · `[~]` in progress (awaiting user test/approval) · `[x]` done & approved

---

## Priority 0 — Loop runs end-to-end and persists one sample (local model)

- [x] **P0.1 — `local` extra + `LocalClient`** (`llm/local.py`, `pyproject.toml`, `llm/__init__.py`)
  HF `transformers` backend satisfying the `LLMClient` protocol. Lazy-load model+tokenizer
  once; build prompt via `apply_chat_template` (`return_dict=True`); run blocking
  `model.generate` inside `asyncio.to_thread` guarded by an `asyncio.Lock`; decode only new
  tokens; return `LLMResponse(text, model, usage, raw)`; forward `**kwargs` to `generate`.
  `ClaudeClient` stays a stub. Tested live by the user against Qwen2.5-7B-Instruct.
- [x] **P0.2 — Prompt helpers + JSON parsing** (`prompting.py`) — `extract_json`/`parse_json`/
  `PromptParseError` + `as_user`, `json_only_instruction`, `describe_feedback`. (Orphaned
  context-renderers and `parse_model` were pruned after the SwitchLingua pivot.)
- [x] **P0.3 — `GenerationAgent.generate`** (`agents/generation.py`). Uses SwitchLingua
  `DATA_GENERATION_PROMPT` (placeholders filled from the request; `education_level`/
  `news_article`/`mcp_result` from `tool_context` or blank); parses `{topic, instances}` and
  takes `instances[0]`; `metadata.topic` from the request; appends refinement feedback on
  retry rounds. No translation (removed from `CSSample`).
- [x] **P0.4 — Four scorer agents** (`agents/scorers.py`) — Fluency / Naturalness / CSRatio /
  SocialCulture. Prompts adapted from SwitchLingua (`core/prompt.py`): each agent's own role
  prompt + JSON schema; shared `_DimensionScorer` maps the dimension's score field →
  `AgentScore.score`, summary/diagnostics → rationale.
- [~] **P0.5 — `SummarizeAgent.summarize`** (`agents/summarize.py`) — deterministic weighted mean
  of the scorers' 0–10 scores; `passed = final >= threshold`; equal weights by default
  (optional `weights` map by agent name); no model call.
- [ ] **P0.6 — Reference `SampleStore`** (`storage/file_store.py`) — JSONL/file store implementing
  the protocol; export from `storage/__init__.py`. No model.
- [ ] **P0.7 — `AcceptanceAgent.accept`** (`agents/acceptance.py`) — attach scores+provenance,
  persist via store. Also fold a **flattened snapshot** of the request's character + basic
  settings + code-switching spec into `metadata` (the internal request has no downstream
  visibility once generated, so denormalize it onto the accepted sample here).

## Priority 1 — Close the refinement loop and make it usable

- [ ] **P1.1 — `RefinerAgent.refine`** (`agents/refiner.py`). Use SwitchLingua `REFINER_PROMPT`
  (`{summary}` = the scorers' rationales). Note the mismatch: SwitchLingua's refiner outputs
  *refined text* directly, but our `EditorAgent.refine` returns `RefinementFeedback` that the
  generator then regenerates from — resolve how to fit (e.g. carry the refined text as the
  feedback `suggestions`, or have generation reuse it).
- [ ] **P1.2 — Seed default `LinguisticPrinciples`** (`principles.py`).
- [ ] **P1.3 — Config + runnable example** (`examples/`, README) — `LocalClient` config surface
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
- Existing loop tests stay green: `pytest` (`tests/test_pipeline_flow.py`) after every step.
- Per-step: user runs the new piece live against the local model and confirms behavior.
- After P0: live end-to-end smoke — `build_default_pipeline(LocalClient(...), FileSampleStore(path))`
  → `await pipeline.run(request)` returns a `CSSample` (or `None`) and writes an accepted sample
  with scores+provenance. Use a smaller model / CPU on the dev box if 7B doesn't fit.
