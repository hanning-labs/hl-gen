"""GenerationAgent — the Generator (Table B.1)."""

from __future__ import annotations

from pydantic import BaseModel

from ..models import CSSample, GenerationContext
from ..prompting import as_user, build_context_block, json_only_instruction, parse_model
from .base import GeneratorAgent

_SYSTEM = (
    "You are an expert bilingual speaker and linguist who writes natural, "
    "authentic code-switched utterances for a speech dataset. You follow the "
    "code-switching type, discourse function, and target embedding ratio "
    "precisely, respect the stated linguistic principles, and produce language a "
    "real speaker with the given persona would actually say."
)

# The JSON shape the model is asked to return.
_RESPONSE_SHAPE = (
    '{"text": "<the code-switched utterance in the matrix language>", '
    '"translation": "<a plain monolingual translation of the utterance>"}'
)


class _GeneratedUtterance(BaseModel):
    """Parsed model output for one generation attempt."""

    text: str
    translation: str | None = None


class GenerationAgent(GeneratorAgent):
    """Produces a CS sentence given topic, persona, and matrix/embedded languages."""

    name = "GenerationAgent"

    async def generate(self, ctx: GenerationContext) -> CSSample:
        instruction = (
            "Write ONE code-switched utterance that matches the specification "
            "above. Keep the matrix (L1) language dominant and embed the second "
            "(L2) language to hit the target ratio, type, and discourse function. "
            "Make it natural and idiomatic for the given persona and setting."
        )
        prompt = (
            f"{build_context_block(ctx)}\n\n"
            f"{instruction}\n\n"
            f"{json_only_instruction(_RESPONSE_SHAPE)}"
        )

        response = await self.llm.complete(as_user(prompt), system=_SYSTEM)
        out = parse_model(response.text, _GeneratedUtterance)

        return CSSample(
            text=out.text,
            translation=out.translation,
            request=ctx.request,
            metadata={"generator": self.name, "refined": ctx.feedback is not None},
        )
