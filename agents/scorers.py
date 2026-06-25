"""Scorer agents (Table B.1): Fluency, Naturalness, CSRatio, SocialCulture.

Each scores one quality dimension on a 0–10 scale; the SummarizeAgent combines
them into S_final. The per-dimension prompts are adapted from the SwitchLingua
project (https://github.com/Shelton1013/SwitchLingua, ``core/prompt.py``): each
agent defines a ``criteria`` tuple of boolean keys. The judge answers true/false
per criterion; the score is ``passed/total * 10``. The shared path in
:class:`_DimensionScorer` fills prompt placeholders, validates the reply, and
maps onto :class:`AgentScore` (rationale from the ``notes`` field).
"""

from __future__ import annotations

import logging

from models import AgentScore, CSSample
from prompting import PromptParseError, as_user, json_only_instruction
from .base import ScorerAgent

log = logging.getLogger(__name__)

_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences. Use true/false for all criterion fields."


# --- SwitchLingua evaluation prompts (verbatim role text + placeholders) ----- #

FLUENCY_PROMPT = (
    "You are **FluencyAgent**. Evaluate the grammatical correctness and syntactic "
    "coherence of the following code-switched text:\n\n"
    "{data_generation_result}\n\n"
    "For each criterion below, answer true or false. Add a \"notes\" field with 1–2 sentences.\n"
    "- free_morpheme_ok: No switching between bound and free morphemes (Poplack 1980)\n"
    "- equivalence_ok: Switches occur where syntactic structures align across languages\n"
    "- word_order_ok: No unnatural mixing of word orders\n\n"
    "Always answer in English in your report."
)

NATURALNESS_PROMPT = (
    "You are **NaturalnessAgent**. Evaluate how natural and authentic the following "
    "code-switched text sounds from a *bilingual speaker's perspective*:\n\n"
    "{data_generation_result}\n\n"
    "For each criterion below, answer true or false. Add a \"notes\" field with 1–2 sentences.\n"
    "- intersentential_ok: Intersentential switching (between sentences) is natural\n"
    "- intrasentential_ok: Intrasentential switching (within a sentence) sounds authentic\n"
    "- bilingual_authentic: Overall sounds like a real bilingual speaker would say it\n\n"
    "Always answer in English in your report."
)

SOCIAL_CULTURAL_PROMPT = (
    "You are **SocioCulturalAgent**. Evaluate whether the following code-switched text "
    "respects cultural norms and uses correct borrowed vocabulary:\n\n"
    "{data_generation_result}\n\n"
    "For each criterion below, answer true or false. Add a \"notes\" field with 1–2 sentences.\n"
    '- borrowed_vocab_correct: Borrowed vocabulary is used correctly (e.g., Cantonese "士多啤梨" for strawberry)\n'
    "- culturally_appropriate: Text respects cultural norms of both languages\n"
    "- register_consistent: Register is consistent with the context\n\n"
    "Always answer in English in your report."
)


class _DimensionScorer(ScorerAgent):
    """Shared scoring path for a SwitchLingua-style single-dimension judge."""

    #: SwitchLingua role prompt; uses {data_generation_result} (and {cs_ratio}).
    prompt: str = ""
    #: Boolean criterion keys the model must answer true/false for.
    criteria: tuple[str, ...] = ()

    def _format_prompt(self, sample: CSSample) -> str:
        body = self.prompt.format(
            data_generation_result=sample.text,
            cs_ratio=sample.request.code_switching.ratio,
        )
        shape = "{" + ", ".join(f'"{k}": true/false' for k in self.criteria) + ', "notes": "..."}'
        return f"{body}\n\n{json_only_instruction(shape)}"

    async def score(self, sample: CSSample) -> AgentScore:
        criteria = self.criteria
        name = self.name

        def _validate(data: object) -> dict:
            if not isinstance(data, dict):
                raise PromptParseError(f"{name}: expected a JSON object, got {type(data).__name__}")
            for k in criteria:
                if k not in data:
                    raise PromptParseError(f"{name}: missing criterion {k!r} in model reply")
            return data

        data = await self._complete_with_retry(as_user(self._format_prompt(sample)), system=_SYSTEM, validate=_validate)
        passed = sum(bool(data[k]) for k in self.criteria)
        score = passed / len(self.criteria) * 10
        log.debug("%s score=%.1f (%d/%d) for sample %r", self.name, score, passed, len(self.criteria), sample.text[:40])
        return AgentScore(agent=self.name, score=score, rationale=data.get("notes", ""))


class FluencyAgent(_DimensionScorer):
    """Verifies grammaticality and the absence of broken morphemes."""

    name = "FluencyAgent"
    prompt = FLUENCY_PROMPT
    criteria = ("free_morpheme_ok", "equivalence_ok", "word_order_ok")


class NaturalnessAgent(_DimensionScorer):
    """Estimates pragmatic plausibility from a bilingual speaker's perspective."""

    name = "NaturalnessAgent"
    prompt = NATURALNESS_PROMPT
    criteria = ("intersentential_ok", "intrasentential_ok", "bilingual_authentic")


class CSRatioAgent(ScorerAgent):
    """Checks whether the token-level language ratio matches the user target.

    Deterministic: uses Unicode script ranges to count tokens per language —
    no LLM call, no parse retries, no hallucinated counts.
    """

    name = "CSRatioAgent"

    def __init__(
        self,
        llm=None,  # accepted for pipeline compatibility; never used
        *,
        name: str | None = None,
        weight: float | None = None,
        parse_retries: int | None = None,
    ) -> None:
        super().__init__(llm, name=name, weight=weight, parse_retries=parse_retries)

    async def score(self, sample: CSSample) -> AgentScore:
        from utils.cs_ratio import count_tokens, l2_presence_score

        l1_lang = sample.request.character.first_language
        l2_lang = sample.request.character.second_language

        l1_count, l2_count = count_tokens(sample.text, l1_lang, l2_lang)

        if l1_count + l2_count == 0:
            log.warning("CSRatioAgent: no tokens detected in %r", sample.text[:40])
            return AgentScore(
                agent=self.name,
                score=0.0,
                rationale=f"No tokens detected (L1={l1_lang}, L2={l2_lang})",
            )

        computed_score = l2_presence_score(l2_count)
        rationale = f"L1={l1_count} tokens, L2={l2_count} tokens"
        log.debug("CSRatioAgent score=%.1f l2_count=%d", computed_score, l2_count)
        return AgentScore(agent=self.name, score=computed_score, rationale=rationale)


class SocialCultureAgent(_DimensionScorer):
    """Validates register, borrowed lexicon, and cultural appropriateness."""

    name = "SocialCultureAgent"
    prompt = SOCIAL_CULTURAL_PROMPT
    criteria = ("borrowed_vocab_correct", "culturally_appropriate", "register_consistent")
