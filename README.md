# hl-gen — multi-pipeline text synthesis framework

[![Write Up — work/hl-gen](https://img.shields.io/badge/Write%20Up-work%2Fhl--gen-cf8b57?style=flat-square&labelColor=1c2126)](https://hanninglabs.com/work/hl-gen)

A closed-loop, multi-agent framework that **generates, scores, and refines text** using an OpenAI-compatible LLM server (e.g. vLLM). Two pipelines ship out of the box:

| Pipeline | What it produces | Entry point |
|---|---|---|
| **Code-switching** | Bilingual utterances scored for fluency, naturalness, CS ratio, and socio-cultural fit | `build_default_pipeline()` in `orchestrator.py` |
| **English topics** | News-grounded prose in a requested style scored for topic relevance, coherence, depth, and style adherence | `build_topics_pipeline()` in `orchestrator.py` |

Both share the same closed loop, agent role hierarchy, storage layer, and tool infrastructure. Adding a third pipeline means writing prompts and subclassing four base classes.


---

## The loop

![The hl-gen closed loop: tool providers fetch() into ArticleSelector, which feeds GenerationAgent; generate() fans out to N concurrent Scorers; SummarizeAgent reduces them to S_final (weighted mean); if score >= threshold the sample goes to AcceptanceAgent and SampleStore (samples.jsonl), otherwise RefinerAgent sends feedback back to GenerationAgent for up to max_refinement_rounds.](docs/loop.png)

Every agent role maps to a base class in `agents/base.py`. Shared logic (`score()`, `refine()`) lives in `DimensionScorer` and `RefinerBase`; pipeline-specific code lives only in prompts and prompt-formatting methods.

---

## Quick start

```bash
pip install -e ".[dev,tools]"
```

For a runnable example see `examples/run_batch.py`. Request shapes are defined in `config.py` (`SynthesisRequest`) and `config_topics.py` (`TopicsRequest`).

### Batch run

```bash
python examples/run_batch.py configs/runs/cs_default.yaml               # code-switching
python examples/run_topics_batch.py configs/runs/topics_default.yaml    # topics
```

Both runners resume from partial output, run up to `max_concurrent` pipelines at once (default 8 — the server's continuous batching absorbs bursts), and write a performance profile to `<output>/profile_*.json`.

### Inference

Inference runs against any OpenAI-compatible server via `OpenAICompatClient` (vLLM recommended; SGLang etc. work identically). Continuous batching, paged KV cache, and OOM handling live server-side.

```bash
scripts/serve_vllm.sh                                # serve on the local GPU (own venv, one-time install)
python examples/run_batch.py configs/runs/cs_vllm.yaml
```

The server runs in its own venv/Docker, so its torch pin never conflicts with the app environment. To run against a VM, start the server there and set `client.base_url: http://<vm>:8000/v1` — nothing else changes.

### News grounding (Currents + NewsAPI)

```bash
echo "CURRENTS_API_KEY=your_key" >> .env
echo "NEWS_API_KEY=your_key" >> .env
```

Pass `tools=[CurrentsTool(), NewsAPITool()]` to either builder (see `tools/currents.py`, `tools/newsapi.py`). Tools are fetched concurrently, and the `ArticleSelector` agent picks the most relevant article across their results and proposes a framing before generation.

---

## Building a new pipeline

Four subclasses + one wiring call. The existing pipelines are the canonical examples — read them alongside these instructions.

### 1. Scorer — `agents/base.py: DimensionScorer`

Subclass `DimensionScorer`. Set `name`, `prompt`, and `criteria` (tuple of boolean key names). Implement `_format_prompt(sample) -> str` to format the prompt with the fields your pipeline uses. See `agents/topics/scorers.py: _TopicDimensionScorer` for the minimal pattern.

`DimensionScorer` provides `score()`: validates all criteria keys are present, computes `passed/total * 10`, returns `AgentScore`. You only write `_format_prompt`.

### 2. Refiner — `agents/base.py: RefinerBase`

Subclass `RefinerBase`. Set `name` and implement `_fill_prompt(sample, report) -> str`. Call `self._summarize_scores(report)` (inherited) to get the sorted score block. See `agents/topics/refiner.py: TopicRefinerAgent` for the minimal pattern.

`RefinerBase` provides `_summarize_scores()` and `refine()` (calls LLM, parses and coerces the JSON). You only write `_fill_prompt`.

### 3. Generator — `agents/base.py: GeneratorAgent`

Subclass `GeneratorAgent`. Implement `generate(ctx) -> CSSample`. Use `ctx.request` for request fields, `ctx.tool_context` for injected news/frame context, and `ctx.feedback` (set on refinement rounds) to append revision guidance. See `agents/topics/generation.py: TopicGenerationAgent`.

### 4. Wire into `orchestrator.py`

Add a `build_my_pipeline(llm, store, *, tools=None) -> SynthesisPipeline` function following the pattern of `build_topics_pipeline()`. `AcceptanceAgent` and `SummarizeAgent` are always reused. The article selector can be reused or replaced.

---

## Agent role reference

| Base class | Implement | Provided | Where |
|---|---|---|---|
| `GeneratorAgent` | `generate(ctx) -> CSSample` | `_complete_with_retry()` | `agents/base.py` |
| `DimensionScorer` | `_format_prompt(sample) -> str` | `score()` | `agents/base.py` |
| `ReducerAgent` | `summarize(scores, threshold) -> ScoreReport` | — | `agents/base.py` |
| `RefinerBase` | `_fill_prompt(sample, report) -> str` | `_summarize_scores()`, `refine()` | `agents/base.py` |
| `SinkAgent` | `accept(sample, report, tool_context)` | — | `agents/base.py` |
| `Agent` | `.select(ctx)` (by convention) | `_complete_with_retry()` | `agents/base.py` |

All agents inherit `_complete_with_retry()`: calls the LLM, parses JSON, retries up to `parse_retries` times (default 2) on parse failure.

---

## Config reference

Configuration is composed from four groups. A run config (`configs/runs/*.yaml`) holds the **pipeline settings** flat at the top level, plus one entry per group — each either the bare name of a variant file (`api: currents` → `configs/api/currents.yaml`) or an inline mapping. `run_config.compose_run_config()` resolves the references; the pipeline's root model validates the result.

Full field definitions live in `run_config.py` (group models), `batch.py` (`BatchConfig`), and `config_topics.py` (`TopicsBatchConfig`). Working examples: everything under `configs/runs/`.

**Pipeline settings** (flat, both pipelines): `n`, `output` (base artifacts directory), `run_name` (subdirectory of `output` holding this run's `samples.jsonl` + `profile_*.json`), `seed`, `score_threshold`, `max_refinement_rounds`, `max_concurrent`.

**`client` group** (`configs/client/`): `backend: openai` → `base_url`, `model`, `max_new_tokens`, `api_key`, `timeout`.

**`api` group** (`configs/api/`): `sources` (which news tools to use), `categories` (news filter — also the topic pool requests sample from), `news_types`, `max_articles`.

**`linguistics` group** (`configs/linguistics/`): code-switching — `language_pairs`, `cs_types`, `cs_functions`, `cs_ratio_min/max`, `age_min/max`, `genders`, `perspectives`, `tenses`, `conversation_types`; topics — `styles`, `perspectives`, `tenses`.

---

## Storage

`FileSampleStore` (see `storage/file_store.py`) writes one JSON line per accepted sample: `{id, saved_at, sample, report}`. It auto-creates parent directories and is append-only and thread-safe.

To use a different backend, implement the `SampleStore` protocol in `storage/base.py`: one async method `save(sample, report) -> str` returning a stable ID.

---

## Tools

Two news providers ship, both taking `api_key`, `max_articles`, `language`, `categories`, and `news_types` in their constructors:

- `CurrentsTool` (see `tools/currents.py`) — Currents API; results under `tool_context["currents_api"]`.
- `NewsAPITool` (see `tools/newsapi.py`) — NewsAPI.org; results under `tool_context["newsapi"]`.

To add a tool, implement the `ToolProvider` protocol in `tools/base.py`: a `name` attribute and an async `fetch(ctx) -> dict`. Multiple tools are fetched concurrently; each result is keyed by `tool.name` in `tool_context`.

---

## Dev

```bash
pip install -e ".[dev,tools]"
pytest tests/                                              # unit tests, mock agents, no LLM
python examples/run_batch.py configs/runs/cs_vllm.yaml     # live smoke run (needs a running server)
```

**Logging** — key loggers: `orchestrator`, `agents.base`, `batch`. Run scripts configure `basicConfig(level=INFO)`.

## Boundaries

- Produces **scored text + metadata**. ASR / WER evaluation and TTS / recording are out of scope.
- `semantic_units` / `utterances` corpus mapping is deferred to a later pass.
