"""TopicRefinerAgent — Editor for the topics pipeline."""

from __future__ import annotations

from models import CSSample, RefinementFeedback, ScoreReport
from prompting import PromptParseError, as_user, json_only_instruction
from agents.base import EditorAgent

_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences."

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


class TopicRefinerAgent(EditorAgent):
    name = "TopicRefinerAgent"

    @staticmethod
    def _summarize_scores(report: ScoreReport) -> str:
        lines = [
            f"- {s.agent}: {s.score:.1f}/10 — {s.rationale or '(no notes)'}"
            for s in sorted(report.scores, key=lambda s: s.score)
        ]
        return "\n".join(lines) if lines else "(no scorer feedback available)"

    def _fill_prompt(self, sample: CSSample, report: ScoreReport) -> str:
        body = TOPIC_REFINER_PROMPT.format(
            text=sample.text,
            summary=self._summarize_scores(report),
        )
        return f"{body}\n\n{json_only_instruction(_RESPONSE_SHAPE)}"

    async def refine(self, sample: CSSample, report: ScoreReport) -> RefinementFeedback:
        name = self.name

        def _validate(data: object) -> dict:
            if not isinstance(data, dict):
                raise PromptParseError(f"{name}: expected a JSON object, got {type(data).__name__}")
            return data

        data = await self._complete_with_retry(as_user(self._fill_prompt(sample, report)), system=_SYSTEM, validate=_validate)

        failures = data.get("failures") or []
        if isinstance(failures, str):
            failures = [failures]
        elif not isinstance(failures, list):
            failures = [str(failures)]
        failures = [str(f) for f in failures if f]

        suggestions = data.get("suggestions") or ""
        if not isinstance(suggestions, str):
            suggestions = str(suggestions)

        return RefinementFeedback(failures=failures, suggestions=suggestions)
