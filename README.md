# code_switch — LinguaMaster code-switching synthesis

A closed-loop, multi-agent pipeline that **generates, scores, and refines
code-switched (CS) text** for Hanning Labs' HL-Code corpus.

## The loop

```
Input Parameters + Linguistic Principles + Tool context (Currents API news)
        │
        ▼
  GenerationAgent ──► [FluencyAgent, NaturalnessAgent, CSRatioAgent, SocialCultureAgent]
        ▲                                   │  (scored concurrently)
        │                                   ▼
   RefinerAgent ◄── no ── score>threshold? ◄── SummarizeAgent (weighted S_final)
                              │ yes
                              ▼
                       AcceptanceAgent ──► FileSampleStore (JSONL)
```

## Quick start

```bash
pip install -e ".[dev,local,tools]"   # installs transformers, torch, pyyaml, requests, …
```

### Single request

```bash
python examples/run_local.py          # writes accepted sample to out/samples.jsonl
```

```python
from orchestrator import build_default_pipeline
from storage.file_store import FileSampleStore
from llm import LocalClient
from config import SynthesisRequest, BasicSetting, CharacterSetting, CodeSwitchingSpec, CodeSwitchType

llm = LocalClient(
    model="Qwen/Qwen2.5-7B-Instruct",  # or "Qwen/Qwen2.5-0.5B-Instruct" for CPU
    device="cpu",                        # "cuda" / "mps" / None → auto
    dtype="auto",
    max_new_tokens=1024,
)
store = FileSampleStore("out/samples.jsonl")
pipeline = build_default_pipeline(llm, store)

request = SynthesisRequest(
    code_switching=CodeSwitchingSpec(type=CodeSwitchType.INTRA_SENTENTIAL, function="expressive", ratio=0.3),
    character=CharacterSetting(first_language="Cantonese", second_language="English", age=28, gender="female"),
    basic=BasicSetting(perspective="first-person", tense="past", topic="movies", conversation_type="casual chat"),
    score_threshold=7.0,
)
sample = await pipeline.run(request)   # CSSample if accepted, else None
```

### Batch run

Define parameter ranges in a YAML config and let the driver randomly sample requests:

```bash
cp configs/default.yaml configs/my_run.yaml  # edit as needed
python examples/run_batch.py configs/my_run.yaml
```

```yaml
# configs/default.yaml (excerpt)
n: 20
output: out/batch_samples.jsonl
score_threshold: 7.0

language_pairs:
  - [Cantonese, English]
  - [Spanish, English]
topics: [technology, movies, food, work, family, travel]
cs_ratio_min: 0.2
cs_ratio_max: 0.5
```

Concurrency is auto-detected from GPU VRAM (~1 pipeline per 8 GB); override with `max_concurrent:` in the config.

### News grounding (Currents API)

```bash
echo "CURRENTS_API_KEY=your_key" >> .env
```

```python
from tools import CurrentsTool
pipeline = build_default_pipeline(llm, store, tools=[CurrentsTool()])
```

Each request fetches relevant articles and injects them into the generation prompt.

## Robustness

- **Parse-failure retries** — agents retry LLM calls up to `parse_retries` times (default 2) when the reply fails to parse as valid JSON. Configurable per-agent.
- **OOM recovery** — `LocalClient` catches `torch.cuda.OutOfMemoryError`, clears the CUDA cache, and retries once before raising a descriptive `RuntimeError`.
- **Logging** — all pipeline stages emit structured log records via Python's `logging` module (`orchestrator`, `agents.base`, `agents.generation`, `agents.scorers`, `llm.local`, `batch`). Run scripts configure `basicConfig(level=INFO)`.

## Dev

```bash
pip install -e ".[dev,local,tools]"
pytest                          # loop control-flow tests (mock agents)
python test.py                  # manual verification for the current step
python examples/run_local.py    # live end-to-end smoke run
```

## Boundaries

- Produces **scored CS text + metadata**. ASR / WER evaluation and TTS / recording are out of scope.
- `semantic_units` / `utterances` corpus mapping is deferred to a later pass.
