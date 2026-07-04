"""TopicRefinerAgent — Editor for the topics pipeline."""

from __future__ import annotations

from models import CSSample, ScoreReport
from prompting import json_only_instruction
from agents.base import RefinerBase, load_guide

TOPIC_REFINER_PROMPT = load_guide("prompts/refiner_topics.md")

_RESPONSE_SHAPE = '{"failures": ["..."], "suggestions": "..."}'


class TopicRefinerAgent(RefinerBase):
    name = "TopicRefinerAgent"

    def _fill_prompt(self, sample: CSSample, report: ScoreReport) -> str:
        req = sample.request
        article = (sample.metadata or {}).get("article") or ""
        article_block = f"- Source news article:\n{article}" if article else "- (No source article.)"
        body = TOPIC_REFINER_PROMPT.format(
            text=sample.text,
            summary=self._summarize_scores(report),
            topic=req.topic,
            style=req.style,
            perspective=req.perspective,
            tense=req.tense,
            article_block=article_block,
        )
        return f"{body}\n\n{json_only_instruction(_RESPONSE_SHAPE)}"
