"""RefinerAgent — the Editor (Table B.1).

The refiner prompt is adapted from the SwitchLingua project
(https://github.com/Shelton1013/SwitchLingua, ``core/prompt.py`` —
``REFINER_PROMPT``). ``{summary}`` is filled with the scorers' rationales (each
judge's 0–10 score plus notes, lowest first) so the model focuses on the weakest
dimensions.

**Deviation from SwitchLingua.** Their refiner emits *refined text* directly. Our
loop instead routes feedback back through the generator: ``EditorAgent.refine``
returns :class:`~models.RefinementFeedback` (``failures`` +
``suggestions``), which :class:`~agents.generation.GenerationAgent`
already appends to its prompt on retry rounds (via ``describe_feedback``). So this
refiner produces *actionable guidance*, not finished text — keeping a single
generation path and one place that owns producing the sample.
"""

from __future__ import annotations

from models import CSSample, RefinementFeedback, ScoreReport
from prompting import PromptParseError, as_user, json_only_instruction
from .base import EditorAgent

_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences."

# SwitchLingua REFINER_PROMPT, adapted: {summary} carries the scorers' rationales,
# {data_generation_result} the failing text. Output is structured feedback (see the
# module docstring for why we diverge from SwitchLingua's direct-text refiner).
REFINER_PROMPT = """\
You are **RefinerAgent**, the editor in a code-switched text generation loop. A
previous attempt was judged on four dimensions (Fluency, Naturalness, CS-Ratio,
Socio-Cultural) and did not pass. Turn the judges' evaluations into concrete,
actionable guidance the generator can use to revise the text on its next attempt.

Code-switched text under review:
{data_generation_result}

Evaluator summary (each judge's score out of 10 and notes, weakest first):
{summary}

Produce a single JSON object:
- "failures": an array of short strings, each naming one concrete problem to fix,
  grounded in the lowest-scoring dimensions above.
- "suggestions": a short paragraph of specific rewrite guidance — what to change
  and how — so the next attempt scores higher while keeping the requested
  languages, code-switching ratio, persona, and topic."""

_RESPONSE_SHAPE = '{"failures": ["..."], "suggestions": "..."}'


class RefinerAgent(EditorAgent):
    """Receives failure explanations and rewrites / re-prompts the generator."""

    name = "RefinerAgent"

    @staticmethod
    def _summarize_scores(report: ScoreReport) -> str:
        """Render the scorers' rationales as the prompt's ``{summary}`` block."""
        lines = [
            f"- {s.agent}: {s.score:.1f}/10 — {s.rationale or '(no notes)'}"
            for s in sorted(report.scores, key=lambda s: s.score)
        ]
        return "\n".join(lines) if lines else "(no scorer feedback available)"

    def _fill_prompt(self, sample: CSSample, report: ScoreReport) -> str:
        body = REFINER_PROMPT.format(
            data_generation_result=sample.text,
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
        if isinstance(failures, str):  # tolerate a single string instead of a list
            failures = [failures]
        elif not isinstance(failures, list):
            failures = [str(failures)]
        failures = [str(f) for f in failures if f]

        suggestions = data.get("suggestions") or ""
        if not isinstance(suggestions, str):
            suggestions = str(suggestions)

        return RefinementFeedback(failures=failures, suggestions=suggestions)
