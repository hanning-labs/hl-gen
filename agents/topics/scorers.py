"""Scorer agents for the English topics pipeline.

Four dimensions: topic relevance, coherence, human-likeness, style adherence.
All inherit _TopicDimensionScorer which omits the CS-specific cs_ratio placeholder.
"""

from __future__ import annotations

import logging

from models import CSSample
from prompting import json_only_instruction
from agents.base import DimensionScorer, _SCORER_SYSTEM, load_guide

log = logging.getLogger(__name__)

_HUMANIZER_GUIDE = load_guide("system/humanizer.md")
_COHERENCE_GUIDE = load_guide("system/coherence.md")
_STYLE_ADHERENCE_GUIDE = load_guide("system/style_adherence.md")
_TOPIC_RELEVANCE_GUIDE = load_guide("system/topic_relevance.md")


class _TopicDimensionScorer(DimensionScorer):
    """Topic-specific prompt formatter: injects {text}, {topic}, {style},
    {perspective}, {tense}, and {article_block} (prompts use what they need)."""

    def _format_prompt(self, sample: CSSample) -> str:
        article = (sample.metadata or {}).get("article") or ""
        article_block = (
            f"Source news article:\n{article}" if article else "(No source article available.)"
        )
        body = self.prompt.format(
            text=sample.text,
            topic=sample.request.topic,
            style=sample.request.style,
            perspective=sample.request.perspective,
            tense=sample.request.tense,
            article_block=article_block,
        )
        shape = "{" + ", ".join(f'"{k}": true/false' for k in self.criteria) + ', "notes": "..."}'
        return f"{body}\n\n{json_only_instruction(shape)}"


RELEVANCE_PROMPT = load_guide("prompts/topic_relevance.md")
COHERENCE_PROMPT = load_guide("prompts/coherence.md")
HUMAN_LIKENESS_PROMPT = load_guide("prompts/human_likeness.md")
STYLE_PROMPT = load_guide("prompts/style_adherence.md")


class TopicRelevanceAgent(_TopicDimensionScorer):
    name = "TopicRelevanceAgent"
    prompt = RELEVANCE_PROMPT
    criteria = ("addresses_topic", "key_concepts_present", "stays_on_topic", "grounded_in_article")
    system = f"{_TOPIC_RELEVANCE_GUIDE}\n\n{_SCORER_SYSTEM}"


class CoherenceAgent(_TopicDimensionScorer):
    name = "CoherenceAgent"
    prompt = COHERENCE_PROMPT
    criteria = ("logical_flow", "clear_transitions", "no_contradictions")
    system = f"{_COHERENCE_GUIDE}\n\n{_SCORER_SYSTEM}"


class HumanLikenessAgent(_TopicDimensionScorer):
    name = "HumanLikenessAgent"
    prompt = HUMAN_LIKENESS_PROMPT
    criteria = ("no_ai_vocab", "natural_rhythm", "no_chatbot_artifacts", "no_formulaic_framing")
    system = f"{_HUMANIZER_GUIDE}\n\n{_SCORER_SYSTEM}"


class StyleAdherenceAgent(_TopicDimensionScorer):
    name = "StyleAdherenceAgent"
    prompt = STYLE_PROMPT
    criteria = ("structure_conforms", "tone_appropriate", "perspective_correct", "tense_correct")
    system = f"{_STYLE_ADHERENCE_GUIDE}\n\n{_SCORER_SYSTEM}"
