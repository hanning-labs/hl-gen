"""RefinerAgent — the Editor (Table B.1).

The refiner prompt is adapted from the SwitchLingua project
(https://github.com/Shelton1013/SwitchLingua, ``core/prompt.py`` —
``REFINER_PROMPT``). ``{summary}`` is filled with the scorers' rationales (each
judge's 0–10 score plus notes, lowest first) so the model focuses on the weakest
dimensions.

**Deviation from SwitchLingua project.** Their refiner emits *refined text* directly. Our
loop instead routes feedback back through the generator: ``EditorAgent.refine``
returns :class:`~models.RefinementFeedback` (``failures`` +
``suggestions``), which :class:`~agents.code_switch.generation.GenerationAgent`
already appends to its prompt on retry rounds (via ``describe_feedback``). So this
refiner produces *actionable guidance*, not finished text — keeping a single
generation path and one place that owns producing the sample.
"""

from __future__ import annotations

from agents.base import RefinerBase
from models import CSSample, ScoreReport
from prompting import json_only_instruction

# hl-gen REFINER_PROMPT, adapted: {summary} carries the scorers' rationales,
# {data_generation_result} the failing text. Output is structured feedback (see the
# module docstring for why we diverge from SwitchLingua project's direct-text refiner).
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


class RefinerAgent(RefinerBase):
    """Receives failure explanations and rewrites / re-prompts the generator."""

    name = "RefinerAgent"

    def _fill_prompt(self, sample: CSSample, report: ScoreReport) -> str:
        body = REFINER_PROMPT.format(
            data_generation_result=sample.text,
            summary=self._summarize_scores(report),
        )
        return f"{body}\n\n{json_only_instruction(_RESPONSE_SHAPE)}"
