"""Manual verification for the current step. Run:  python test.py

End-to-end pipeline smoke test (live, loads the local model): wire the default
pipeline and run one request through generate -> score -> summarize ->
accept/refine. Prints the outcome and, if accepted, the persisted record.
"""

import asyncio

from code_switch import (
    BasicSetting,
    CharacterSetting,
    CodeSwitchingSpec,
    CodeSwitchType,
    SynthesisRequest,
    build_default_pipeline,
)
from code_switch.llm import LocalClient
from code_switch.storage import FileSampleStore

# max_refinement_rounds kept low for a quicker smoke run; lower score_threshold
# if you want to make the accept (persist) path easier to hit with a local model.
REQUEST = SynthesisRequest(
    code_switching=CodeSwitchingSpec(
        type=CodeSwitchType.INTRA_SENTENTIAL, function="emphasis", ratio=0.3
    ),
    character=CharacterSetting(
        first_language="Cantonese", second_language="English", age=28, gender="female"
    ),
    basic=BasicSetting(
        perspective="first-person",
        tense="past",
        topic="movies",
        conversation_type="casual chat",
    ),
    score_threshold=7.0,
    max_refinement_rounds=2,
)


async def main() -> None:
    # Swap in a smaller model if 7B won't fit, e.g. model="Qwen/Qwen2.5-0.5B-Instruct".
    llm = LocalClient()
    store = FileSampleStore("out/pipeline_samples.jsonl")
    pipeline = build_default_pipeline(llm, store)

    print("=== running pipeline ===")
    sample = await pipeline.run(REQUEST)

    if sample is None:
        print("result: REJECTED (no attempt cleared the threshold within the rounds)")
        return

    print("result: ACCEPTED")
    print("text       :", sample.text)
    print("final_score:", sample.metadata.get("final_score"))
    print("passed     :", sample.metadata.get("passed"))
    print("scores     :", sample.metadata.get("scores"))

    record = store.read_all()[-1]
    print("\n=== persisted record ===")
    print("id        :", record["id"])
    print("saved_at  :", record["saved_at"])
    print("spec      :", record["sample"]["metadata"]["spec"])


if __name__ == "__main__":
    asyncio.run(main())
