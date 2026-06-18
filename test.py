import asyncio
from code_switch.llm import LocalClient
from code_switch.agents.scorers import FluencyAgent, NaturalnessAgent, CSRatioAgent, SocialCultureAgent
from code_switch.models import CSSample
from code_switch import SynthesisRequest, CodeSwitchingSpec, CodeSwitchType, CharacterSetting, BasicSetting

req = SynthesisRequest(
    code_switching=CodeSwitchingSpec(type=CodeSwitchType.INTRA_SENTENTIAL, function="emphasis", ratio=0.3),
    character=CharacterSetting(first_language="Cantonese", second_language="English", age=28, gender="female"),
    basic=BasicSetting(perspective="first-person", tense="past", topic="movies", conversation_type="casual chat"),
)
sample = CSSample(text="我哋琴晚去睇咗 a really good movie。", translation="We saw a really good movie last night.", request=req)
llm = LocalClient()
for Agent in (FluencyAgent, NaturalnessAgent, CSRatioAgent, SocialCultureAgent):
    s = asyncio.run(Agent(llm).score(sample))
    print(s.agent, s.score, "—", s.rationale)
