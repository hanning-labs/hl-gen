# hl-gen — multi-pipeline text synthesis framework

A closed-loop, multi-agent framework that **generates, scores, and refines text** using local LLMs. Two pipelines ship out of the box:

| Pipeline | What it produces | Entry point |
|---|---|---|
| **Code-switching** | Bilingual utterances scored for fluency, naturalness, CS ratio, and socio-cultural fit | `build_default_pipeline()` in `orchestrator.py` |
| **English topics** | News-grounded prose in a requested style scored for topic relevance, coherence, depth, and style adherence | `build_topics_pipeline()` in `orchestrator.py` |

Both share the same closed loop, agent role hierarchy, storage layer, and tool infrastructure. Adding a third pipeline means writing prompts and subclassing four base classes.

---

## The loop

```
Tool providers (CurrentsTool, …)
        │  fetch()
        ▼
ArticleSelector  ──►  GenerationAgent
                             │  generate()
                             ▼
             ┌── Scorers (concurrent) ──┐
             │  score() × N dimensions  │
             └──────────────────────────┘
                             │
                        SummarizeAgent
                        (S_final = weighted mean)
                             │
                    score >= threshold?
                      ├─ yes ──► AcceptanceAgent ──► SampleStore (JSONL)
                      └─ no  ──► RefinerAgent ──► feedback ──► GenerationAgent
                                 (up to max_refinement_rounds)
```

Every agent role maps to a base class in `agents/base.py`. Shared logic (`score()`, `refine()`) lives in `DimensionScorer` and `RefinerBase`; pipeline-specific code lives only in prompts and prompt-formatting methods.

---

## Quick start

```bash
pip install -e ".[dev,local,tools]"
```

For a runnable single-request example see `examples/run_local.py`. Request shapes are defined in `config.py` (`SynthesisRequest`) and `config_topics.py` (`TopicsRequest`).

### Batch run

```bash
python examples/run_batch.py configs/default.yaml          # code-switching
python examples/run_topics_batch.py configs/topics_default.yaml   # topics
```

Both runners resume from partial output, auto-detect GPU concurrency (~1 pipeline per 8 GB VRAM), and write a performance profile to `docs/`.

### News grounding (Currents API)

```bash
echo "CURRENTS_API_KEY=your_key" >> .env
```

Pass `tools=[CurrentsTool()]` to either builder (see `tools/currents.py`). The `ArticleSelector` agent picks the most relevant article and proposes a framing before generation.

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

Full field definitions live in `batch.py` (`BatchConfig`, `LocalClientConfig`) and `config_topics.py` (`TopicsBatchConfig`). Working examples: `configs/default.yaml` and `configs/topics_default.yaml`.

**Shared keys** (both pipelines): `n`, `output`, `seed`, `score_threshold`, `max_refinement_rounds`, `max_concurrent`, `client` (model, device, dtype, max\_new\_tokens, max\_batch\_size, compile\_model).

**Code-switching only**: `language_pairs`, `cs_types`, `cs_functions`, `cs_ratio_min/max`, `age_min/max`, `genders`, `conversation_types`.

**Topics only**: `styles`, `perspectives`, `tenses`.

**CurrentsTool filters** (both): `categories`, `news_types`.

---

## Storage

`FileSampleStore` (see `storage/file_store.py`) writes one JSON line per accepted sample: `{id, saved_at, sample, report}`. It auto-creates parent directories and is append-only and thread-safe.

To use a different backend, implement the `SampleStore` protocol in `storage/base.py`: one async method `save(sample, report) -> str` returning a stable ID.

---

## Tools

`CurrentsTool` (see `tools/currents.py`) fetches news from the Currents API and returns articles under `tool_context["currents_api"]`. Constructor accepts `api_key`, `max_articles`, `language`, `categories`, and `news_types`.

To add a tool, implement the `ToolProvider` protocol in `tools/base.py`: a `name` attribute and an async `fetch(ctx) -> dict`. Multiple tools are fetched concurrently; each result is keyed by `tool.name` in `tool_context`.

---

## Dev

```bash
pip install -e ".[dev,local,tools]"
pytest tests/                          # unit tests, mock agents, no LLM
python examples/run_local.py           # live end-to-end smoke run
```

**Logging** — key loggers: `orchestrator`, `agents.base`, `agents.generation`, `agents.scorers`, `llm.local`, `batch`. Run scripts configure `basicConfig(level=INFO)`.

**OOM recovery** — `LocalClient` catches `torch.cuda.OutOfMemoryError`, clears the CUDA cache, and retries once before raising.

## Boundaries

- Produces **scored text + metadata**. ASR / WER evaluation and TTS / recording are out of scope.
- `semantic_units` / `utterances` corpus mapping is deferred to a later pass.
