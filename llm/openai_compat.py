"""OpenAI-compatible server implementation of :class:`LLMClient`.

Talks to any OpenAI-compatible ``/v1/chat/completions`` endpoint — a local
vLLM server by default (see ``scripts/serve_vllm.sh``), but equally SGLang or
a server on a remote VM: only ``base_url`` changes. The server owns batching
(continuous batching), KV-cache management, and OOM recovery, so there is no
client-side queue or drain loop — each :meth:`complete` is a single awaited
HTTP call.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from .base import LLMResponse, Message

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "AxionML/Qwen3.5-9B-NVFP4"
DEFAULT_MAX_NEW_TOKENS = 1024


class OpenAICompatClient:
    """:class:`LLMClient` backed by an OpenAI-compatible chat-completions server.

    Parameters
    ----------
    base_url:
        Server endpoint, e.g. ``http://localhost:8000/v1`` (local vLLM) or a
        remote VM's address.
    model:
        Model id as the server knows it (for vLLM, the HF id it was launched
        with). Per-call ``complete(model=...)`` overrides are passed through;
        the server rejects ids it isn't serving.
    max_new_tokens:
        Default generation length, sent as ``max_tokens``; override per call
        via ``complete(..., max_new_tokens=N)``.
    api_key:
        Bearer token; local vLLM/SGLang servers ignore it, any placeholder works.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        api_key: str = "EMPTY",
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.max_new_tokens = max_new_tokens
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return a completion for ``messages`` under an optional ``system`` prompt.

        ``max_new_tokens`` is mapped to the API's ``max_tokens``; ``json_schema``
        is mapped to ``response_format`` for server-side constrained decoding.
        Remaining ``**kwargs`` (``temperature``, ``top_p``, …) pass through unchanged.
        """
        chat: list[dict[str, str]] = []
        if system is not None:
            chat.append({"role": "system", "content": system})
        chat.extend({"role": m.role, "content": m.content} for m in messages)

        request: dict[str, Any] = {
            "model": model or self.model,
            "messages": chat,
            "max_tokens": kwargs.pop("max_new_tokens", self.max_new_tokens),
        }
        json_schema = kwargs.pop("json_schema", None)
        if json_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        request.update(kwargs)

        response = await self._client.chat.completions.create(**request)
        message = response.choices[0].message
        text = message.content or ""

        # vLLM's --reasoning-parser splits the trace out server-side (field is
        # `reasoning` in vLLM 0.24, `reasoning_content` in older/other servers);
        # without it, split an inline <think> block as a fallback.
        reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or None
        )
        if reasoning is None and "</think>" in text:
            reasoning, _, text = text.partition("</think>")
            reasoning = reasoning.removeprefix("<think>").strip()
            text = text.strip()

        usage: dict[str, Any] = {}
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return LLMResponse(
            text=text, model=response.model, reasoning=reasoning, usage=usage, raw=None
        )
