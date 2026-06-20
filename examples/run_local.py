"""End-to-end smoke run against the local model.

Usage
-----
    python examples/run_local.py

The script builds the default pipeline with a LocalClient and a FileSampleStore,
runs one synthesis request, and prints the result. Accepted samples are written
to ``out/samples.jsonl``.

LocalClient config surface
--------------------------
    model           HF model id           default: "Qwen/Qwen2.5-7B-Instruct"
    device          device_map value       default: None → "auto"
                    ("cpu", "cuda", "mps")
    dtype           torch_dtype            default: "auto"
                    ("float16", "bfloat16")
    max_new_tokens  generation length      default: 1024

Swap in a smaller model for quick iteration on a CPU-only box:
    llm = LocalClient(model="Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
"""

import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BasicSetting,
    CharacterSetting,
    CodeSwitchingSpec,
    CodeSwitchType,
    SynthesisRequest,
)
from llm import LocalClient
from orchestrator import build_default_pipeline
from storage.file_store import FileSampleStore

REQUEST = SynthesisRequest(
    code_switching=CodeSwitchingSpec(
        type=CodeSwitchType.INTRA_SENTENTIAL,
        function="expressive",
        ratio=0.3,
    ),
    character=CharacterSetting(
        first_language="Cantonese",
        second_language="English",
        age=28,
        gender="female",
    ),
    basic=BasicSetting(
        perspective="first-person",
        tense="past",
        topic="movies",
        conversation_type="casual chat",
    ),
    score_threshold=7.0,
    max_refinement_rounds=3,
)


async def main() -> None:
    llm = LocalClient()  # override: LocalClient(model="...", device="cpu")
    store = FileSampleStore("out/samples.jsonl")
    pipeline = build_default_pipeline(llm, store)

    print("Running synthesis pipeline…")
    print(f"  model:    {llm.model}")
    print(f"  topic:    {REQUEST.basic.topic}")
    print(f"  L1/L2:    {REQUEST.character.first_language} / {REQUEST.character.second_language}")
    print(f"  cs_type:  {REQUEST.code_switching.type.value}")
    print(f"  threshold:{REQUEST.score_threshold}  max_rounds:{REQUEST.max_refinement_rounds}")
    print()

    sample = await pipeline.run(REQUEST)

    if sample is None:
        print("No sample reached the score threshold within the allowed rounds.")
        return

    print("=== Accepted sample ===")
    print(sample.text)
    print()
    print("=== Metadata ===")
    for k, v in sample.metadata.items():
        if k != "instances":
            print(f"  {k}: {v}")
    print()
    print(f"Written to: {store.path}")


if __name__ == "__main__":
    asyncio.run(main())
