"""Manual smoke test for the generator (and scorers) against the local model.

Run from the code_switch/ directory (or anywhere the package is installed):

    python test.py

Loads the local model once, generates one code-switched sample from a request,
prints it, then runs the four scorers on what was generated.
"""

import asyncio

from code_switch import (
    BasicSetting,
    CharacterSetting,
    CodeSwitchingSpec,
    CodeSwitchType,
    SynthesisRequest,
)
from code_switch.agents.generation import GenerationAgent
from code_switch.agents.scorers import (
    CSRatioAgent,
    FluencyAgent,
    NaturalnessAgent,
    SocialCultureAgent,
)
from code_switch.llm import LocalClient
from code_switch.models import GenerationContext

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
    # Swap in a smaller model if 7B won't fit, e.g.:
    # llm = LocalClient(model="Qwen/Qwen2.5-0.5B-Instruct")
    llm = LocalClient()

    print("=== generating ===")
    gen = GenerationAgent(llm)
    sample = await gen.generate(GenerationContext(request=REQUEST))
    print("text       :", sample.text)
    print("topic      :", sample.metadata.get("topic"))
    print("instances  :", sample.metadata.get("instances"))

    print("\n=== scoring the generated sample ===")
    for agent_cls in (FluencyAgent, NaturalnessAgent, CSRatioAgent, SocialCultureAgent):
        score = await agent_cls(llm).score(sample)
        print(f"{score.agent:18} {score.score:>5} — {score.rationale}")


if __name__ == "__main__":
    asyncio.run(main())
