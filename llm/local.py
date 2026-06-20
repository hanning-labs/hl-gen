"""Local Hugging Face ``transformers`` implementation of :class:`LLMClient`.

This is the primary backend for the synthesis framework: it runs a chat model
in-process via ``transformers``. The heavy dependencies (``transformers``,
``torch``, ``accelerate``) live behind the ``local`` optional dependency group
and are imported lazily, so the package stays importable without them.

``model.generate`` is blocking, and a single loaded model instance is not safe
for concurrent generation. The orchestrator runs the scorer agents with
``asyncio.gather``, so :meth:`complete` offloads generation to a worker thread
(``asyncio.to_thread``) guarded by an :class:`asyncio.Lock`. Concurrent callers
therefore serialize through the one loaded model — correct for a single model on
a single device — while the event loop stays unblocked.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .base import LLMResponse, Message

log = logging.getLogger(__name__)

# Default local model. Override per instance; the 14B variant
# ("Qwen/Qwen2.5-14B-Instruct") is a drop-in if you have the memory for it.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 1024


class LocalClient:
    """:class:`LLMClient` backed by a local Hugging Face ``transformers`` model.

    The model and tokenizer are loaded once, on first :meth:`complete`, and
    cached on the instance.

    Parameters
    ----------
    model:
        Hugging Face model id to load.
    device:
        ``device_map`` passed to ``from_pretrained`` (e.g. ``"cpu"``, ``"mps"``,
        ``"cuda"``). ``None`` uses ``"auto"``.
    dtype:
        ``torch_dtype`` passed to ``from_pretrained`` (e.g. ``"auto"``,
        ``"float16"``, ``"bfloat16"``).
    max_new_tokens:
        Default generation length; override per call via ``complete(..., max_new_tokens=N)``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: str = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any = None
        self._model: Any = None
        self._lock = asyncio.Lock()

    def _ensure_loaded(self) -> None:
        """Load the tokenizer and model once. Blocking — call from a worker thread."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "LocalClient requires the 'local' optional dependencies. "
                'Install them with: pip install -e ".[local]"'
            ) from exc

        log.info("Loading model %r (device=%s, dtype=%s)", self.model, self.device or "auto", self.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=self.dtype,
            device_map=self.device or "auto",
        )

    def _run(
        self,
        messages: list[Message],
        system: str | None,
        gen_kwargs: dict[str, Any],
    ) -> LLMResponse:
        """Blocking generation path. Runs inside ``asyncio.to_thread``."""
        import torch

        self._ensure_loaded()
        tokenizer, model = self._tokenizer, self._model

        # If a JSON schema was requested, build a prefix_allowed_tokens_fn that
        # constrains generation to tokens valid under that schema at every step.
        json_schema = gen_kwargs.pop("_json_schema", None)
        if json_schema is not None:
            try:
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_transformers_prefix_allowed_tokens_fn,
                )
                gen_kwargs["prefix_allowed_tokens_fn"] = (
                    build_transformers_prefix_allowed_tokens_fn(
                        tokenizer, JsonSchemaParser(json_schema)
                    )
                )
            except ImportError as exc:
                raise ImportError(
                    "Constrained JSON decoding requires lm-format-enforcer. "
                    'Install it with: pip install -e ".[local]"'
                ) from exc

        chat: list[dict[str, str]] = []
        if system is not None:
            chat.append({"role": "system", "content": system})
        chat.extend({"role": m.role, "content": m.content} for m in messages)

        inputs = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        if gen_kwargs.get("pad_token_id") is None and tokenizer.pad_token_id is None:
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

        t0 = time.monotonic()
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.warning("CUDA OOM — clearing cache and retrying once")
            try:
                with torch.inference_mode():
                    generated = model.generate(**inputs, **gen_kwargs)
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(
                    "CUDA out of memory after cache clear. "
                    "Try a smaller model, reduce max_new_tokens, or use device='cpu'."
                ) from exc
        log.debug("generate() took %.2fs", time.monotonic() - t0)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = generated[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return LLMResponse(
            text=text,
            model=self.model,
            usage={
                "prompt_tokens": int(prompt_len),
                "completion_tokens": int(new_tokens.shape[-1]),
                "total_tokens": int(generated.shape[-1]),
            },
            raw=None,
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return a completion for ``messages`` under an optional ``system`` prompt.

        ``model`` is accepted for protocol compatibility but the local backend
        serves the single model loaded at construction; requesting a different
        one raises ``ValueError`` (reload by constructing a new client instead).
        ``**kwargs`` are forwarded to ``model.generate`` (e.g. ``temperature``,
        ``do_sample``, ``top_p``); ``max_new_tokens`` defaults to the instance value.
        """
        if model is not None and model != self.model:
            raise ValueError(
                f"LocalClient is loaded with {self.model!r}; per-call model "
                f"override to {model!r} is not supported. Construct a new "
                "LocalClient for a different model."
            )

        # json_schema is our own kwarg — intercept it before forwarding the rest
        # to model.generate(). Pass it through to _run via a private key so it
        # stays out of the HuggingFace generate() call signature.
        json_schema = kwargs.pop("json_schema", None)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, **kwargs}
        if json_schema is not None:
            gen_kwargs["_json_schema"] = json_schema

        async with self._lock:
            return await asyncio.to_thread(self._run, messages, system, gen_kwargs)
