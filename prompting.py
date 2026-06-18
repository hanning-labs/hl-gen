"""Shared prompt helpers and JSON-response parsing.

Agents build their prompts from the SwitchLingua templates and parse the local
model's free-form reply back into JSON. Local models have no guaranteed
JSON-schema mode, so the parsing here is deliberately forgiving — it strips code
fences and locates the JSON span inside surrounding prose.
"""

from __future__ import annotations

import json
from typing import Any

from llm.base import Message
from models import RefinementFeedback

__all__ = [
    "PromptParseError",
    "json_only_instruction",
    "as_user",
    "describe_feedback",
    "extract_json",
    "parse_json",
]


class PromptParseError(ValueError):
    """Raised when a model reply can't be parsed into the expected JSON shape."""


def json_only_instruction(example: str) -> str:
    """A standard 'reply with only this JSON shape' instruction for agent prompts."""
    return (
        "Respond with a single JSON object and nothing else — no prose, no "
        f"markdown, no code fences. Use exactly this shape:\n{example}"
    )


def as_user(content: str) -> list[Message]:
    """Wrap a string as the single user turn expected by ``LLMClient.complete``."""
    return [Message(role="user", content=content)]


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
