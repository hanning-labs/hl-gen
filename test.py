"""Manual verification for the current step. Run:  python test.py

P1.1 — RefinerAgent.refine. Feeds the refiner a failing ScoreReport (real
AgentScore data with low scores + rationales — not an LLM double) and runs it
live against the local model. Confirms refine() returns RefinementFeedback whose
`failures`/`suggestions` are grounded in the weakest dimensions, and shows how
that feedback renders into the next generation prompt (describe_feedback).
"""

import asyncio

from config import (
    BasicSetting,
    CharacterSetting,
    CodeSwitchingSpec,
    CodeSwitchType,
    SynthesisRequest,
)
from models import AgentScore, CSSample, ScoreReport
from agents.refiner import RefinerAgent
from llm import LocalClient
from prompting import describe_feedback

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

# A deliberately weak sample + a failing report (scores below a 7.0 threshold on
# two dimensions). These are plain data artifacts, not mocked LLM calls.
SAMPLE = CSSample(
    text="I go cinema yesterday 睇 movie, it was 好 good 啦 the storyline 也 very nice 嘅.",
    request=REQUEST,
)
REPORT = ScoreReport(
    scores=[
        AgentScore(
            agent="FluencyAgent",
            score=3.5,
            rationale="Broken word order; 'I go cinema yesterday' drops articles "
            "and tense; switches violate the Free Morpheme Constraint around 睇/睇.",
        ),
        AgentScore(
            agent="NaturalnessAgent",
            score=4.0,
            rationale="Over-switching makes it read like keyword salad, not how a "
            "Cantonese-English bilingual would actually speak.",
        ),
        AgentScore(
            agent="CSRatioAgent",
            score=6.5,
            rationale="computed_ratio ~55%:45%, English-heavy vs the 30% English target.",
        ),
        AgentScore(
            agent="SocialCultureAgent",
            score=8.0,
            rationale="No culture-specific vocabulary problems.",
        ),
    ],
    final_score=5.5,
    passed=False,
)


async def main() -> None:
    # Swap in a smaller model if 7B won't fit, e.g. model="Qwen/Qwen2.5-0.5B-Instruct".
    refiner = RefinerAgent(LocalClient())

    print("=== sample under review ===")
    print(SAMPLE.text)
    print("\n=== scorer summary fed to the refiner ===")
    print(RefinerAgent._summarize_scores(REPORT))

    print("\n=== running RefinerAgent.refine (live) ===")
    feedback = await refiner.refine(SAMPLE, REPORT)

    print("\n=== RefinementFeedback ===")
    print("failures:")
    for f in feedback.failures:
        print("  -", f)
    print("suggestions:", feedback.suggestions)

    print("\n=== how it renders into the next generation prompt ===")
    print(describe_feedback(feedback))


if __name__ == "__main__":
    asyncio.run(main())
