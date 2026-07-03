"""GenerationAgent — the Generator (Table B.1).

The generation prompt is adapted from the SwitchLingua project
(https://github.com/Shelton1013/SwitchLingua, ``core/prompt.py`` —
``DATA_GENERATION_PROMPT``). Placeholders are filled from our
:class:`~config.SynthesisRequest`; fields SwitchLingua project expects but we
don't model (``education_level``, ``news_article``, ``mcp_result``) are pulled
from ``ctx.tool_context`` when present and otherwise left blank. On refinement
rounds the prior attempt's feedback is appended so the loop can revise.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import GeneratorAgent, fill_template, load_guide
from models import CSSample, GenerationContext
from prompting import PromptParseError, as_user, describe_feedback

log = logging.getLogger(__name__)


def _format_news(news_ctx: Any) -> str:
    """Render a tool_context news value into a prompt-friendly string."""
    if not news_ctx:
        return ""
    if isinstance(news_ctx, str):
        return news_ctx
    if not isinstance(news_ctx, dict):
        return str(news_ctx)
    articles = news_ctx.get("articles") or []
    if not articles:
        return ""
    lines = []
    for a in articles:
        title = a.get("title", "").strip()
        body = a.get("body", "").strip()
        if title:
            lines.append(f"- {title}")
            if body:
                lines.append(f"  {body}")
    return "\n".join(lines)

_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences."

# hl-gen DATA_GENERATION_PROMPT (role text + {placeholders}). Example-output
# braces are single (not the doubled langchain form) since we fill by targeted
# replacement, not str.format.
DATA_GENERATION_PROMPT = load_guide("prompts/data_generation.md")


class GenerationAgent(GeneratorAgent):
    """Produces a CS sentence given topic, persona, and matrix/embedded languages."""

    name = "GenerationAgent"

    def _fill_prompt(self, ctx: GenerationContext) -> str:
        req = ctx.request
        cs, ch, b = req.code_switching, req.character, req.basic
        tc = ctx.tool_context or {}
        values = {
            "first_language": ch.first_language,
            "second_language": ch.second_language,
            "cs_function": cs.function,
            "cs_type": cs.type.value,
            "cs_ratio": cs.ratio,
            "tense": b.tense,
            "perspective": b.perspective,
            "gender": ch.gender,
            "age": ch.age,
            "education_level": tc.get("education_level", "unspecified"),
            "news_article": _format_news({"articles": [tc["selected_article"]]})
            if tc.get("selected_article")
            else _format_news(tc.get("news")),
            "interaction_frame": tc.get("interaction_frame", ""),
            "conversation_type": b.conversation_type,
            "topic": b.topic,
            "mcp_result": tc.get("mcp", ""),
        }
        prompt = fill_template(DATA_GENERATION_PROMPT, **values)

        if ctx.feedback is not None:
            prompt = (
                f"{prompt}\n\n{describe_feedback(ctx.feedback)}\n\n"
                "Revise the code-switched text to address the feedback above, "
                "keeping the same JSON output format."
            )
        return prompt

    async def generate(self, ctx: GenerationContext) -> CSSample:
        def _validate(data: object) -> dict:
            if not isinstance(data, dict):
                raise PromptParseError(f"generation: expected a JSON object, got {type(data).__name__}")
            instances = data.get("instances")
            if not isinstance(instances, list) or not instances:
                raise PromptParseError("generation reply had no non-empty 'instances' array")
            return data

        data = await self._complete_with_retry(as_user(self._fill_prompt(ctx)), system=_SYSTEM, validate=_validate)
        instances = data["instances"]
        llm_resp = getattr(self, "_last_llm_response", None)

        log.debug("%s generated %d instances for topic=%r", self.name, len(instances), ctx.request.basic.topic)
        first = instances[0]
        if isinstance(first, str):
            text = first
        elif isinstance(first, dict):  # multi-turn message pair
            text = first.get("content") or first.get("text") or json.dumps(first, ensure_ascii=False)
        else:
            text = str(first)

        return CSSample(
            text=text,
            request=ctx.request,
            metadata={
                "generator": self.name,
                "refined": ctx.feedback is not None,
                "topic": ctx.request.basic.topic,
                "instances": instances,
                "llm_model": llm_resp.model if llm_resp else None,
                "llm_usage": llm_resp.usage if llm_resp else None,
            },
        )
