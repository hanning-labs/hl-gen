# code_switch — LinguaMaster code-switching synthesis

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

## Usage

```bash
pip install -e ".[dev,local]"   # installs transformers, torch, accelerate
python examples/run_local.py    # end-to-end run; writes to out/samples.jsonl
```

```python
from orchestrator import build_default_pipeline
from storage.file_store import FileSampleStore
from llm import LocalClient
from config import SynthesisRequest, ...

# LocalClient config surface:
#   model           HF model id           default: "Qwen/Qwen2.5-7B-Instruct"
#   device          device_map            default: None → "auto" ("cpu", "cuda", "mps")
#   dtype           torch_dtype           default: "auto" ("float16", "bfloat16")
#   max_new_tokens  generation length     default: 1024
llm = LocalClient(model="Qwen/Qwen2.5-7B-Instruct", device="cpu")
store = FileSampleStore("out/samples.jsonl")

pipeline = build_default_pipeline(llm, store)
sample = await pipeline.run(request)  # CSSample if accepted, else None
```

## Boundaries

- Produces **scored CS text + metadata**. The SwitchLingua benchmark
  (ASR / WER eval) and real TTS / recording are **out of scope** here.
- Accepted `CSSample`s can later be mapped into the planned audio corpus DB
  (`semantic_units` / `utterances`; see `.claude/memory/audio-db-schema.md`).

## Dev

```bash
pip install -e ".[dev,local]"   # pydantic + pytest + transformers + torch
pytest                          # loop control-flow tests (mock agents)
python examples/run_local.py    # live end-to-end smoke run
```
