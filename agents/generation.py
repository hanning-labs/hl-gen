"""GenerationAgent — the Generator (Table B.1).

The generation prompt is adapted from the SwitchLingua project
(https://github.com/Shelton1013/SwitchLingua, ``core/prompt.py`` —
``DATA_GENERATION_PROMPT``). Placeholders are filled from our
:class:`~config.SynthesisRequest`; fields SwitchLingua expects but we
don't model (``education_level``, ``news_article``, ``mcp_result``) are pulled
from ``ctx.tool_context`` when present and otherwise left blank. On refinement
rounds the prior attempt's feedback is appended so the loop can revise.
"""

from __future__ import annotations

import json
from typing import Any

from models import CSSample, GenerationContext
from prompting import PromptParseError, as_user, describe_feedback, parse_json
from .base import GeneratorAgent


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

# SwitchLingua DATA_GENERATION_PROMPT (role text + {placeholders}). Example-output
# braces are single (not the doubled langchain form) since we fill by targeted
# replacement, not str.format.
DATA_GENERATION_PROMPT = """\
You are a multilingual generation agent. You generate code-switched text based on
the user's instructions. Follow these guidelines:

1. Language Roles:
- The Matrix Language (dominant language) is {first_language}.
- The Embedded Language (secondary language) is {second_language}.

2. Code-Switching Functions:
- Directive: Include or exclude certain listeners.
- Expressive: Show identity, cultural connection, or emotion.
- Referential: If a concept is easier to express in the other language.
- Phatic: Repeat or emphasize by switching languages.
- Metalinguistic: Quoting or commenting on a phrase in the other language.
- Poetic: Jokes or wordplay in the embedded language.
- The function is {cs_function}

3. Code-Switching Types:
- Intersentential: Switch languages across sentence boundaries. The switch occurs at sentence or clause boundaries. The speaker finishes one sentence (or clause) in Language A, then starts the next sentence (or clause) in Language B. This form often appears when the speaker wants to address different audiences or emphasize particular parts of the conversation. It can be used for directive functions (e.g., to include/exclude certain listeners), or for phatic emphasis of entire sentences.
  - Examples: English to Spanish, "I have a big project due tomorrow. ¿Puedes ayudarme?" (English sentence first, then Spanish question.)
  - Examples: Hindi to English, "Maine kal tumhe phone kiya tha. But you didn't pick up!" (Hindi clause followed by an English clause.)
  - Examples: Chinese to English, "今天天氣真的好好。 I think we should go for a walk." (Chinese sentence about the weather, then an English suggestion.)
  - Examples: Filipino (Tagalog) to English, "Gusto kong kumain sa labas mamaya. Let's try that new restaurant!" (Tagalog statement, then an English invitation.)
  - Use one full sentence in the matrix language, then start a new sentence in the embedded language.
  - Each entire sentence is generally in one language, though small connectors (like "and," "but") may appear.
- Intrasentential: Switch languages within a single sentence.
  - This is often more complex syntactically, because the switch must respect each language's grammar constraints (like subject-verb-object ordering, morphological rules, etc.).
  - Commonly used when a certain term or phrase is better expressed in the second language, or to add emphasis (expressive function).
  - Examples: English to Portuguese, "I don't know o meu lugar nesse mundo." (Partial phrase in Portuguese: "my place in this world.")
  - Examples: Chinese to English, "我老是去那家 coffee shop，因为那里真的很 peaceful，而且vibe也不错。" (Chinese sentence, then English statements.)
  - Within a single sentence, embed a short phrase or clause in the second language (e.g., for an object, an adjective, or a common expression).
  - Remind the model to maintain grammatical coherence; e.g., do not place an English determiner in a position that violates the word order rules of the main language.
- Extra-sentential / Tag switching: A short tag, filler, or interjection from the second language is inserted into an otherwise single-language utterance. Common examples are "right?", "you know?" or discourse markers like "anyway," "well," "deshou?", "baka," etc.
  - Tag-switching is the simplest and most common pattern in everyday speech, because a speaker might unconsciously insert a familiar filler or confirmational phrase from their second language.
  - Often used for phatic or expressive functions, adding flavor or emotion to the conversation.
  - Examples: English (main) + Japanese (tag), "It's a good movie, deshou?"
  - Examples: Chinese (main) + English (tag), "好辛苦呀, oh my gosh!"
- The type is {cs_type}

4. Ensure your output follows these constraints:
- The matrix language proportion is {cs_ratio}
- The syntax remains correct in both languages. (Observe free morpheme constraint & equivalence constraint.)
- Make it sound natural to bilingual speakers (avoid unnatural mixing).
- Respect socio-cultural norms (correct borrowed words, e.g., Chinese might use '士多啤梨' instead of '草莓').

5. Output must be in JSON format with keys: [topic, instances].
- 'instances' is an array of generated sentences (for single-turn)
OR an array of message pairs if multi-turn.

6. Language Requirements:
- Tense: {tense}
- Perspective: {perspective}

7. Persona:
- Gender: {gender}
- Age: {age}
- Education Level: {education_level}

8. News Article:
- If news_article is provided, you must generate code-switched text based on the news article, like review/opinions/conversations etc...
- News Article: {news_article}

9. The conversation type is {conversation_type}

**Example Output Structure format** (for a multi-turn example in Cantonese-English Mixed Language):
{
    "instances": [
    "XXXXX？",
    "XXXXX！",
    "XXXXX。",
    "XXXXX。"
    ],
}

Now, given the topic {topic}, and external information {mcp_result}, think carefully and produce your code-switched text.

### INTERNAL (do NOT reveal):
1. Parse the {first_language} sentence into a dependency tree.
2. Translate it into {second_language}.
3. Align tokens between the two sentences.
4. Locate all switchable spans that satisfy the Equivalence
    & Functional-Head constraints; pick the best one.
- Keep all intermediate notes private.
### END INTERNAL"""


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
            "news_article": _format_news(tc.get("news")),
            "conversation_type": b.conversation_type,
            "topic": b.topic,
            "mcp_result": tc.get("mcp", ""),
        }
        prompt = DATA_GENERATION_PROMPT
        for key, value in values.items():
            prompt = prompt.replace("{" + key + "}", str(value))

        if ctx.feedback is not None:
            prompt = (
                f"{prompt}\n\n{describe_feedback(ctx.feedback)}\n\n"
                "Revise the code-switched text to address the feedback above, "
                "keeping the same JSON output format."
            )
        return prompt

    async def generate(self, ctx: GenerationContext) -> CSSample:
        response = await self.llm.complete(as_user(self._fill_prompt(ctx)), system=_SYSTEM)
        data = parse_json(response.text)
        if not isinstance(data, dict):
            raise PromptParseError(f"generation: expected a JSON object, got {type(data).__name__}")

        instances = data.get("instances")
        if not isinstance(instances, list) or not instances:
            raise PromptParseError("generation reply had no non-empty 'instances' array")

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
            },
        )
