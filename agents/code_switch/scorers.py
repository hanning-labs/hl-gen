"""Scorer agents (Table B.1): Fluency, Naturalness, CSRatio, SocialCulture.

Each scores one quality dimension on a 0–10 scale; the SummarizeAgent combines
them into S_final. The per-dimension prompts are adapted from the SwitchLingua
project (https://github.com/Shelton1013/SwitchLingua, ``core/prompt.py``): each
agent defines a ``criteria`` tuple of boolean keys. The judge answers true/false
per criterion; the score is ``passed/total * 10``. Shared scoring logic lives in
:class:`~agents.base.DimensionScorer`; subclasses here only define prompts and
``_format_prompt``.
"""

from __future__ import annotations

import logging

from agents.base import DimensionScorer, ScorerAgent, _SCORER_SYSTEM, load_guide
from models import AgentScore, CSSample
from prompting import json_only_instruction

log = logging.getLogger(__name__)

FLUENCY_PROMPT = load_guide("prompts/fluency.md")
NATURALNESS_PROMPT = load_guide("prompts/naturalness.md")
SOCIAL_CULTURAL_PROMPT = load_guide("prompts/social_culture.md")
_SOCIAL_CULTURE_GUIDE = load_guide("system/social_culture.md")


class _CSDimensionScorer(DimensionScorer):
    """CS-specific prompt formatter: injects {data_generation_result} and {cs_ratio}."""

    def _format_prompt(self, sample: CSSample) -> str:
        body = self.prompt.format(
            data_generation_result=sample.text,
            cs_ratio=sample.request.code_switching.ratio,
        )
        shape = "{" + ", ".join(f'"{k}": true/false' for k in self.criteria) + ', "notes": "..."}'
        return f"{body}\n\n{json_only_instruction(shape)}"


class FluencyAgent(_CSDimensionScorer):
    """Verifies grammaticality and the absence of broken morphemes."""

    name = "FluencyAgent"
    prompt = FLUENCY_PROMPT
    criteria = ("free_morpheme_ok", "equivalence_ok", "word_order_ok")


class NaturalnessAgent(_CSDimensionScorer):
    """Estimates pragmatic plausibility from a bilingual speaker's perspective."""

    name = "NaturalnessAgent"
    prompt = NATURALNESS_PROMPT
    criteria = ("intersentential_ok", "intrasentential_ok", "bilingual_authentic")


class CSRatioAgent(ScorerAgent):
    """Scores L2 *presence* — that the sample actually code-switches at all.

    Deterministic: counts tokens per language and scores off the L2 count —
    no LLM call, no parse retries, no hallucinated counts.

    **This does not enforce the requested ratio.** Despite the name,
    ``request.code_switching.ratio`` is never compared against; the score is
    ``min(l2_count * 5, 10)`` (see :func:`~utils.cs_ratio.l2_presence_score`),
    so two L2 tokens is a full score regardless of the target. It catches the
    common failure — a "code-switched" sample that is monolingual — and nothing
    finer. Enforcing the ratio itself is future work.
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


class SocialCultureAgent(_CSDimensionScorer):
    """Validates register, borrowed lexicon, and cultural appropriateness."""

    name = "SocialCultureAgent"
    prompt = SOCIAL_CULTURAL_PROMPT
    criteria = ("borrowed_vocab_correct", "culturally_appropriate", "register_consistent")
    system = f"{_SOCIAL_CULTURE_GUIDE}\n\n{_SCORER_SYSTEM}"
