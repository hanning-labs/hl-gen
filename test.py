"""Manual verification for the current step. Run:  python test.py

P0.6 — FileSampleStore (no model needed): save an accepted sample to JSONL and
read it back.
"""

import asyncio

from code_switch import (
    BasicSetting,
    CharacterSetting,
    CodeSwitchingSpec,
    CodeSwitchType,
    SynthesisRequest,
)
from code_switch.models import AgentScore, CSSample, ScoreReport
from code_switch.storage import FileSampleStore, SampleStore

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
)


async def main() -> None:
    store = FileSampleStore("out/samples.jsonl")
    print("is SampleStore:", isinstance(store, SampleStore))

    sample = CSSample(text="我哋去咗 cinema 睇戲。", request=REQUEST)
    report = ScoreReport(
        scores=[AgentScore(agent="FluencyAgent", score=9.0, rationale="clean")],
        final_score=9.0,
        passed=True,
    )

    sample_id = await store.save(sample, report)
    print("saved id:", sample_id)

    records = store.read_all()
    print("record count:", len(records))
    print("last record:", records[-1])


if __name__ == "__main__":
    asyncio.run(main())
