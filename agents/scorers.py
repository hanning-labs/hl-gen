"""Scorer agents (Table B.1): Fluency, Naturalness, CSRatio, SocialCulture.

Each scores one quality dimension on a 0–10 scale; the SummarizeAgent combines
them into S_final. The per-dimension prompts are adapted from the SwitchLingua
project (https://github.com/Shelton1013/SwitchLingua, ``core/prompt.py``): each
agent uses its own role prompt and JSON schema (``fluency_score``/``errors``,
``naturalness_score``/``observations``, ``ratio_score``/``computed_ratio``,
``socio_cultural_score``/``issues``). The shared path in :class:`_DimensionScorer`
fills the prompt's placeholders, parses the JSON reply, and maps the dimension's
score field onto :class:`AgentScore` (rationale from the summary/notes fields).
"""

from __future__ import annotations

import logging

from models import AgentScore, CSSample
from prompting import PromptParseError, as_user, json_only_instruction
from .base import ScorerAgent

log = logging.getLogger(__name__)

_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences."


# --- SwitchLingua evaluation prompts (verbatim role text + placeholders) ----- #

FLUENCY_PROMPT = (
    "You are **FluencyAgent**. Your task is to evaluate the grammatical "
    "correctness and syntactic coherence of code-switched text. Specifically:\n\n"
    "1. **Check for code-switching constraints** from *Poplack (1980)*:\n"
    "- **Free Morpheme Constraint**: no switching between bound and free morphemes.\n"
    "- **Equivalence Constraint**: switches should occur where syntactic structures align.\n\n"
    "2. **Check for grammatical errors** or unnatural mixing of word orders.\n\n"
    "3. **Output**:\n"
    "- A `fluency_score` (0 to 10).\n"
    "- A list of identified `errors` (if any), with `description` and `constraint_violated`\n"
    "- A short `summary` of overall fluency.\n"
    "given the code-switched text {data_generation_result}.\n"
    "Always answer in English in your report."
)

NATURALNESS_PROMPT = (
    "You are **NaturalnessAgent**. Your job is to evaluate how natural and "
    "authentic the code-switched text is from a *bilingual speaker's perspective*:\n\n"
    "1. **Check typical code-switching usage**: Intersentential, Intrasentential, "
    "and Tag Switching patterns.\n"
    "2. **Consider factors from *Auer (1998)***\n"
    "3. **Output**:\n"
    "- A `naturalness_score` (0 to 10).\n"
    "- A list of `observations` about unnatural phrases.\n"
    "- A `summary` describing overall authenticity.\n"
    "given the code-switched text {data_generation_result}.\n"
    "Always answer in English in your report."
)

CS_RATIO_PROMPT = (
    "You are **CSRatioAgent**. Evaluate the *Code-Switching Ratio* by counting "
    "tokens for each language and comparing to desired ratio.\n\n"
    "**Output**:\n"
    "- A `ratio_score` (0 to 10) reflecting target match.\n"
    '- A `computed_ratio` breakdown (e.g., "66% : 34%").\n'
    "- A `notes` field with observations.\n\n"
    "given the desired ratio: {cs_ratio} and text: {data_generation_result}.\n"
    "Always answer in English in your report."
)

SOCIAL_CULTURAL_PROMPT = (
    "You are **SocioCulturalAgent**. Ensure code-switched text respects *cultural "
    "norms* and uses *correct borrowed words*.\n\n"
    '1. **Check culture-specific vocabulary** (e.g., Cantonese "士多啤梨" for strawberry)\n'
    "2. **Output**:\n"
    "- A `socio_cultural_score` (0 to 10).\n"
    "- An array of `issues` if found.\n"
    "- A short `summary` with assessment.\n\n"
    "given the code-switched text {data_generation_result}.\n"
    "Always answer in English in your report."
)


class _DimensionScorer(ScorerAgent):
    """Shared scoring path for a SwitchLingua-style single-dimension judge."""

    #: SwitchLingua role prompt; uses {data_generation_result} (and {cs_ratio}).
    prompt: str = ""
    #: Key holding the 0–10 score in the model's JSON reply.
    score_key: str = "score"
    #: Keys whose values (joined) become the AgentScore rationale.
    rationale_keys: tuple[str, ...] = ()
    #: Example JSON shape appended to enforce a parseable reply.
    response_shape: str = ""

    def _format_prompt(self, sample: CSSample) -> str:
        body = self.prompt.format(
            data_generation_result=sample.text,
            cs_ratio=sample.request.code_switching.ratio,
        )
        return f"{body}\n\n{json_only_instruction(self.response_shape)}"

    def _rationale(self, data: dict) -> str:
        parts: list[str] = []
        for key in self.rationale_keys:
            value = data.get(key)
            if value:
                parts.append(value if isinstance(value, str) else str(value))
        return " — ".join(parts)

    async def score(self, sample: CSSample) -> AgentScore:
        score_key = self.score_key
        name = self.name

        def _validate(data: object) -> dict:
            if not isinstance(data, dict):
                raise PromptParseError(f"{name}: expected a JSON object, got {type(data).__name__}")
            if score_key not in data:
                raise PromptParseError(f"{name}: missing {score_key!r} in model reply")
            try:
                float(data[score_key])
            except (TypeError, ValueError) as exc:
                raise PromptParseError(f"{name}: non-numeric {score_key!r} in model reply") from exc
            return data

        data = await self._complete_with_retry(as_user(self._format_prompt(sample)), system=_SYSTEM, validate=_validate)
        score = max(0.0, min(10.0, float(data[self.score_key])))
        log.debug("%s score=%.1f for sample %r", self.name, score, sample.text[:40])
        return AgentScore(agent=self.name, score=score, rationale=self._rationale(data))


class FluencyAgent(_DimensionScorer):
    """Verifies grammaticality and the absence of broken morphemes."""

    name = "FluencyAgent"
    prompt = FLUENCY_PROMPT
    score_key = "fluency_score"
    rationale_keys = ("summary", "errors")
    response_shape = (
        '{"fluency_score": <0-10>, "errors": [{"description": "...", '
        '"constraint_violated": "..."}], "summary": "..."}'
    )


class NaturalnessAgent(_DimensionScorer):
    """Estimates pragmatic plausibility from a bilingual speaker's perspective."""

    name = "NaturalnessAgent"
    prompt = NATURALNESS_PROMPT
    score_key = "naturalness_score"
    rationale_keys = ("summary", "observations")
    response_shape = '{"naturalness_score": <0-10>, "observations": ["..."], "summary": "..."}'


class CSRatioAgent(_DimensionScorer):
    """Checks whether the token-level language ratio matches the user target."""

    name = "CSRatioAgent"
    prompt = CS_RATIO_PROMPT
    score_key = "ratio_score"
    rationale_keys = ("computed_ratio", "notes")
    response_shape = '{"ratio_score": <0-10>, "computed_ratio": "66% : 34%", "notes": "..."}'


class SocialCultureAgent(_DimensionScorer):
    """Validates register, borrowed lexicon, and cultural appropriateness."""

    name = "SocialCultureAgent"
    prompt = SOCIAL_CULTURAL_PROMPT
    score_key = "socio_cultural_score"
    rationale_keys = ("summary", "issues")
    response_shape = '{"socio_cultural_score": <0-10>, "issues": ["..."], "summary": "..."}'
