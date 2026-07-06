"""Shared prompt helpers and JSON-response parsing.

Agents build their prompts from the SwitchLingua templates and parse the
model's reply back into JSON. Callers that pass ``json_schema`` get
server-side guided decoding (see ``agents.base.DimensionScorer``/``RefinerBase``),
but the parsing here stays deliberately forgiving as a fallback — it strips
code fences and locates the JSON span inside surrounding prose.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from llm.base import Message
from models import RefinementFeedback

__all__ = [
    "PromptParseError",
    "json_only_instruction",
    "as_user",
    "describe_feedback",
    "iter_json_candidates",
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


def _find_json_span(s: str, pos: int = 0) -> tuple[str, int] | None:
    """Return the first balanced ``{...}`` / ``[...]`` span at or after ``pos``,
    respecting string literals, as ``(span, start_index)``."""
    start = next((i for i in range(pos, len(s)) if s[i] in "{["), None)
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
                return s[start : i + 1], start
    return None


def iter_json_candidates(text: str) -> Iterator[Any]:
    """Yield every decodable JSON value from the balanced ``{...}``/``[...]``
    spans in a possibly noisy model reply, in order of appearance.

    A reply may quote the requested shape (``{"text": "..."}``) inside prose or
    a leaked thinking trace before the actual answer, so callers should keep
    consuming candidates until one passes their validation.
    """
    if not text or not text.strip():
        return
    stripped = _strip_code_fences(text)
    for candidate in (stripped,) if stripped == text.strip() else (stripped, text):
        pos = 0
        while (found := _find_json_span(candidate, pos)) is not None:
            span, start = found
            try:
                yield json.loads(span)
            except json.JSONDecodeError:
                pass
            pos = start + 1


def parse_json(text: str) -> Any:
    """Extract and decode JSON from a model reply (first decodable span)."""
    for value in iter_json_candidates(text):
        return value
    raise PromptParseError(f"no decodable JSON object/array in model output: {text[:300]!r}")
