import asyncio
from code_switch.llm import LocalClient
from code_switch.agents.generation import GenerationAgent
from code_switch.models import GenerationContext
from code_switch import SynthesisRequest, CodeSwitchingSpec, CodeSwitchType, CharacterSetting, BasicSetting

req = SynthesisRequest(
    code_switching=CodeSwitchingSpec(type=CodeSwitchType.INTRA_SENTENTIAL, function="emphasis", ratio=0.3),
    character=CharacterSetting(first_language="Cantonese", second_language="English", age=28, gender="female"),
    basic=BasicSetting(perspective="first-person", tense="past", topic="movies", conversation_type="casual chat"),
)
gen = GenerationAgent(LocalClient())          # or a small model for a fast check
sample = asyncio.run(gen.generate(GenerationContext(request=req)))
print(sample.text, "||", sample.translation, "||", sample.metadata)
