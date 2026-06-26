"""TopicRefinerAgent — Editor for the topics pipeline."""

from __future__ import annotations

from models import CSSample, ScoreReport
from prompting import json_only_instruction
from agents.base import RefinerBase

TOPIC_REFINER_PROMPT = """\
You are **TopicRefinerAgent**, the editor in an English topic-content generation loop.
A previous attempt was judged on four dimensions (Topic Relevance, Coherence, Factual Quality,
Style Adherence) and did not pass. Turn the judges' evaluations into concrete, actionable
guidance the generator can use to revise the text on its next attempt.

Text under review:
{text}

Evaluator summary (each judge's score out of 10 and notes, weakest first):
{summary}

Produce a single JSON object:
- "failures": an array of short strings, each naming one concrete problem to fix,
  grounded in the lowest-scoring dimensions above.
- "suggestions": a short paragraph of specific rewrite guidance — what to change
  and how — so the next attempt scores higher while keeping the requested topic, style,
  perspective, and tense."""

_RESPONSE_SHAPE = '{"failures": ["..."], "suggestions": "..."}'


class TopicRefinerAgent(RefinerBase):
    name = "TopicRefinerAgent"

    def _fill_prompt(self, sample: CSSample, report: ScoreReport) -> str:
        body = TOPIC_REFINER_PROMPT.format(
            text=sample.text,
            summary=self._summarize_scores(report),
        )
        return f"{body}\n\n{json_only_instruction(_RESPONSE_SHAPE)}"
