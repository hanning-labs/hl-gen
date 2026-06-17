# data_synthesis — LinguaMaster code-switching synthesis

A closed-loop, multi-agent pipeline that **generates, scores, and refines
code-switched (CS) text** for Hanning Labs' HL-Code corpus. This is the
**skeleton**: the orchestrator loop is real; agent / LLM / tool bodies raise
`NotImplementedError` and are filled in one-by-one in later passes.

## The loop

```
Input Parameters + Linguistic Principles + Tool context
        │
        ▼
  GenerationAgent ──► [FluencyAgent, NaturalnessAgent, CSRatioAgent, SocialCultureAgent]
        ▲                                   │  (scored concurrently)
        │                                   ▼
   RefinerAgent ◄── no ── score>threshold? ◄── SummarizeAgent (weighted S_final)
                              │ yes
                              ▼
                       AcceptanceAgent ──► SampleStore
```

## Diagram → code

| Framework element            | Code                                                    |
| ---------------------------- | ------------------------------------------------------- |
| Input Parameters             | `config.py` (`SynthesisRequest` + sub-specs)            |
| Linguistic Principles        | `principles.py` (`LinguisticPrinciples`)                |
| Tool Integration (MCP)       | `tools/base.py` (`ToolProvider`, `CustomHook`) + stubs  |
| GenerationAgent (Generator)  | `agents/generation.py`                                  |
| Fluency/Naturalness/CSRatio/SocialCulture (Scorers) | `agents/scorers.py`              |
| SummarizeAgent (Reducer)     | `agents/summarize.py`                                   |
| RefinerAgent (Editor)        | `agents/refiner.py`                                     |
| AcceptanceAgent (Sink)       | `agents/acceptance.py` → `storage/base.py`              |
| The refinement loop          | `orchestrator.py` (`SynthesisPipeline.run`)             |
| LLM backend (swappable)      | `llm/base.py` (`LLMClient`) + `llm/claude.py` stub      |

## Usage (once agents are implemented)

```python
from code_switch.data_synthesis import build_default_pipeline, SynthesisRequest
from code_switch.data_synthesis.llm import ClaudeClient

pipeline = build_default_pipeline(ClaudeClient(), my_store, tools=[...])
sample = await pipeline.run(request)  # CSSample if accepted, else None
```

## Boundaries

- Produces **scored CS text + metadata**. The SwitchLingua benchmark
  (ASR / WER eval) and real TTS / recording are **out of scope** here.
- Accepted `CSSample`s can later be mapped into the planned audio corpus DB
  (`semantic_units` / `utterances`; see `.claude/memory/audio-db-schema.md`).

## Dev

```bash
pip install -e ".[dev]"   # pydantic + pytest + pytest-asyncio
pytest                    # loop control-flow tests (mock agents)
```
