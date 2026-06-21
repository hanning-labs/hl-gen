"""TopicGenerationAgent — English-only topic content generator."""

from __future__ import annotations

import logging
from typing import Any

from models import CSSample, GenerationContext
from prompting import PromptParseError, as_user, describe_feedback
from agents.base import GeneratorAgent

log = logging.getLogger(__name__)

_SYSTEM = "Respond with only the requested JSON object — no prose, no code fences."

TOPIC_GENERATION_PROMPT = """\
You are a content generation agent. Write either realistic conversational English text or high quality English text on the given topic, given the context you are provided. You always generate 1 sentence for the content.

Style: {style}
Topic: {topic}
Tense: {tense}
Perspective: {perspective}
{news_block}
Write in the requested style. The text must directly address the topic and read naturally.

Output a single JSON object:
{{"instances": ["<generated text>"]}}"""


def _format_article(tc: dict) -> str:
    if tc.get("selected_article"):
        a = tc["selected_article"]
        title = (a.get("title") or "").strip()
        body = (a.get("body") or "").strip()
        return f"- {title}\n  {body}" if body else f"- {title}"
    articles = (tc.get("news") or {}).get("articles") or []
    if not articles:
        return ""
    a = articles[0]
    title = (a.get("title") or "").strip()
    body = (a.get("body") or "").strip()
    return f"- {title}\n  {body}" if body else f"- {title}"


class TopicGenerationAgent(GeneratorAgent):
    """Generates English topic content grounded in news articles."""

    name = "TopicGenerationAgent"

    def _fill_prompt(self, ctx: GenerationContext) -> str:
        req = ctx.request
        tc = ctx.tool_context or {}
        article_text = _format_article(tc)
        frame = tc.get("interaction_frame", "")

        news_block = ""
        if article_text:
            news_block = f"\nNews article (ground your content in this):\n{article_text}"
            if frame:
                news_block += f"\nEngagement frame: {frame}"

        prompt = TOPIC_GENERATION_PROMPT.format(
            style=req.style,
            topic=req.topic,
            tense=req.tense,
            perspective=req.perspective,
            news_block=news_block,
        )

        if ctx.feedback is not None:
            prompt = (
                f"{prompt}\n\n{describe_feedback(ctx.feedback)}\n\n"
                "Revise the text to address the feedback above, keeping the same JSON output format."
            )
        return prompt

    async def generate(self, ctx: GenerationContext) -> CSSample:
        def _validate(data: Any) -> dict:
            if not isinstance(data, dict):
                raise PromptParseError(f"generation: expected a JSON object, got {type(data).__name__}")
            instances = data.get("instances")
            if not isinstance(instances, list) or not instances:
                raise PromptParseError("generation reply had no non-empty 'instances' array")
            return data

        data = await self._complete_with_retry(as_user(self._fill_prompt(ctx)), system=_SYSTEM, validate=_validate)
        instances = data["instances"]
        llm_resp = getattr(self, "_last_llm_response", None)

        log.debug("%s generated %d instances for topic=%r", self.name, len(instances), ctx.request.topic)
        text = instances[0] if isinstance(instances[0], str) else str(instances[0])

        return CSSample(
            text=text,
            request=ctx.request,
            metadata={
                "generator": self.name,
                "refined": ctx.feedback is not None,
                "topic": ctx.request.topic,
                "style": ctx.request.style,
                "instances": instances,
                "llm_model": llm_resp.model if llm_resp else None,
                "llm_usage": llm_resp.usage if llm_resp else None,
            },
        )
