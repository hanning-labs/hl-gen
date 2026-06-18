"""Shared prompt assembly and JSON-response parsing.

Generator, scorer, and refiner agents all need two things: turn a
:class:`~code_switch.models.GenerationContext` (request + principles + tool
context + refinement feedback) into readable prompt text, and turn a local
model's free-form reply back into a validated object. Local models have no
guaranteed JSON-schema mode, so the parsing here is deliberately forgiving —
it strips code fences, locates the JSON span inside surrounding prose, and
validates against a target Pydantic model.

This module owns the **shared** pieces (context rendering + JSON extraction).
Each agent supplies its own task instruction and the exact response shape it
asks for.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .config import SynthesisRequest
from .llm.base import Message
from .models import GenerationContext, RefinementFeedback
from .principles import LinguisticPrinciples

__all__ = [
    "PromptParseError",
    "describe_request",
    "describe_principles",
    "describe_tool_context",
    "describe_feedback",
    "build_context_block",
    "json_only_instruction",
    "as_user",
    "extract_json",
    "parse_json",
    "parse_model",
]

T = TypeVar("T", bound=BaseModel)


class PromptParseError(ValueError):
    """Raised when a model reply can't be parsed/validated into the target shape."""


# --------------------------------------------------------------------------- #
# Context rendering                                                            #
# --------------------------------------------------------------------------- #


def describe_request(request: SynthesisRequest) -> str:
    """Render the persona, discourse framing, and code-switching spec as text."""
    cs, ch, b = request.code_switching, request.character, request.basic
    return "\n".join(
        [
            "Speaker / character:",
            f"- First language (L1 / matrix): {ch.first_language}",
            f"- Second language (L2 / embedded): {ch.second_language}",
            f"- Age: {ch.age}",
            f"- Gender: {ch.gender}",
            "Discourse setting:",
            f"- Topic: {b.topic}",
            f"- Conversation type: {b.conversation_type}",
            f"- Perspective: {b.perspective}",
            f"- Tense: {b.tense}",
            "Code-switching spec:",
            f"- Type: {cs.type.value}",
            f"- Function: {cs.function}",
            f"- Target L2 (embedded-language) token ratio: {cs.ratio:.2f}",
        ]
    )


def describe_principles(principles: LinguisticPrinciples) -> str:
    """Render the three constraint families as bullet lists; empty families skipped."""
    sections = (
        ("Syntactic constraints", principles.syntactic),
        ("Semantic / discourse constraints", principles.semantic_discourse),
        ("Sociolinguistic constraints", principles.sociolinguistic),
    )
    out: list[str] = []
    for title, items in sections:
        if items:
            out.append(f"{title}:")
            out.extend(f"- {item}" for item in items)
    return "\n".join(out)


def _compact(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def describe_tool_context(tool_context: dict[str, Any]) -> str:
    """Render external tool-provider context, one line per provider; empty -> ''."""
    if not tool_context:
        return ""
    out = ["External context:"]
    out.extend(f"- {name}: {_compact(value)}" for name, value in tool_context.items())
    return "\n".join(out)


def describe_feedback(feedback: RefinementFeedback | None) -> str:
    """Render refinement feedback for a retry round; ``None`` -> '' (first attempt)."""
    if feedback is None:
        return ""
    out = ["Revision feedback from the previous attempt (address every point):"]
    if feedback.failures:
        out.append("Failures:")
        out.extend(f"- {f}" for f in feedback.failures)
    if feedback.suggestions:
        out.append(f"Suggestions: {feedback.suggestions}")
    return "\n".join(out)


def build_context_block(ctx: GenerationContext) -> str:
    """Combine request + principles + tool context + feedback into one prompt block."""
    parts = [describe_request(ctx.request)]
    for section in (
        describe_principles(ctx.request.principles),
        describe_tool_context(ctx.tool_context),
        describe_feedback(ctx.feedback),
    ):
        if section:
            parts.append(section)
    return "\n\n".join(parts)


def json_only_instruction(example: str) -> str:
    """A standard 'reply with only this JSON shape' instruction for agent prompts."""
    return (
        "Respond with a single JSON object and nothing else — no prose, no "
        f"markdown, no code fences. Use exactly this shape:\n{example}"
    )


def as_user(content: str) -> list[Message]:
    """Wrap a string as the single user turn expected by ``LLMClient.complete``."""
    return [Message(role="user", content=content)]


# --------------------------------------------------------------------------- #
# JSON response parsing                                                        #
# --------------------------------------------------------------------------- #


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].startswith("```"):  # drop opening ``` or ```json
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):  # drop closing fence
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _find_json_span(s: str) -> str | None:
    """Return the first balanced ``{...}`` / ``[...]`` span, respecting string literals."""
    start = next((i for i, ch in enumerate(s) if ch in "{["), None)
    if start is None:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def extract_json(text: str) -> str:
    """Extract the JSON object/array substring from a possibly noisy model reply."""
    if not text or not text.strip():
        raise PromptParseError("empty model output")
    span = _find_json_span(_strip_code_fences(text)) or _find_json_span(text)
    if span is None:
        raise PromptParseError(f"no JSON object/array found in model output: {text!r}")
    return span


def parse_json(text: str) -> Any:
    """Extract and decode JSON from a model reply."""
    span = extract_json(text)
    try:
        return json.loads(span)
    except json.JSONDecodeError as exc:
        raise PromptParseError(f"invalid JSON in model output: {exc}") from exc


def parse_model(text: str, model_cls: type[T]) -> T:
    """Extract, decode, and validate a model reply against ``model_cls``."""
    data = parse_json(text)
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise PromptParseError(
            f"model output did not match {model_cls.__name__}: {exc}"
        ) from exc
