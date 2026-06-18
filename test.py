"""Manual verification for the current step. Run:  python test.py

P0.7 — AcceptanceAgent.accept (no model needed): accept a sample, confirm the
flattened spec + scores land in metadata and the record is persisted via the store.
"""

import asyncio

from code_switch import (
    BasicSetting,
    CharacterSetting,
    CodeSwitchingSpec,
    CodeSwitchType,
    SynthesisRequest,
)
from code_switch.agents.acceptance import AcceptanceAgent
from code_switch.models import AgentScore, CSSample, ScoreReport
from code_switch.storage import FileSampleStore

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
    acceptor = AcceptanceAgent(llm=None, store=store)  # accept() never touches the llm

    sample = CSSample(text="我哋去咗 cinema 睇咗 a great movie。", request=REQUEST)
    report = ScoreReport(
        scores=[
            AgentScore(agent="FluencyAgent", score=9.0, rationale="grammatical"),
            AgentScore(agent="CSRatioAgent", score=8.0, rationale="66% : 34%"),
        ],
        final_score=8.5,
        passed=True,
    )

    await acceptor.accept(sample, report)

    print("=== metadata after accept ===")
    for key in ("accepted_by", "final_score", "passed", "scores", "spec"):
        print(f"{key}: {sample.metadata.get(key)}")

    print("\n=== persisted record ===")
    record = store.read_all()[-1]
    print("id:", record["id"])
    print("saved_at:", record["saved_at"])
    print("stored metadata.spec:", record["sample"]["metadata"]["spec"])
    print("stored report.passed:", record["report"]["passed"])


if __name__ == "__main__":
    asyncio.run(main())
