"""Local Hugging Face ``transformers`` implementation of :class:`LLMClient`.

This is the primary backend for the synthesis framework: it runs a chat model
in-process via ``transformers``. The heavy dependencies (``transformers``,
``torch``, ``accelerate``) live behind the ``local`` optional dependency group
and are imported lazily, so the package stays importable without them.

``model.generate`` is blocking and unsafe for concurrent calls. Rather than
serialising through a lock, :meth:`complete` enqueues each request as a
:class:`_BatchItem` with an :class:`asyncio.Future` and returns immediately to
the event loop. A background drain task collects up to ``max_batch_size``
pending items (or fires after ``batch_timeout_sec``), runs one batched
``model.generate()`` in a worker thread, then resolves every Future with its
slice of the output.

With ``max_batch_size=1`` (the default) the drain fires immediately on each
item, giving identical behaviour to the old lock-based approach.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .base import LLMResponse, Message

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 1024


@dataclass
class _BatchItem:
    messages: list[Message]
    system: str | None
    gen_kwargs: dict[str, Any]
    future: asyncio.Future


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
    max_batch_size:
        Maximum number of requests to group into one ``model.generate()`` call.
        ``1`` disables batching (default, safe for all hardware).
    batch_timeout_sec:
        How long the drain task waits for a batch to fill before firing with
        whatever it has. Lower values reduce latency; higher values improve
        batching efficiency under load.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: str = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_batch_size: int = 1,
        batch_timeout_sec: float = 0.02,
        compile_model: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size
        self.batch_timeout_sec = batch_timeout_sec
        self.compile_model = compile_model
        self._tokenizer: Any = None
        self._model: Any = None
        self._queue: asyncio.Queue[_BatchItem | None] = asyncio.Queue()
        self._drain_task: asyncio.Task | None = None

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

        log.info(
            "Loading model %r (device=%s, dtype=%s, compile=%s)",
            self.model, self.device or "auto", self.dtype, self.compile_model,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=self.dtype,
            device_map=self.device or "auto",
        )

        if self.compile_model:
            import torch
            log.info("Compiling model with torch.compile (first batch will be slow) ...")
            self._model = torch.compile(self._model, mode="reduce-overhead")

    def _run_batch(self, items: list[_BatchItem]) -> list[LLMResponse]:
        """Blocking batched generation. Runs inside ``asyncio.to_thread``.

        Left-pads all prompts to the same length so ``model.generate`` receives
        a single batch tensor. Per-item JSON schema constraints are supported via
        a combined ``prefix_allowed_tokens_fn`` that routes by ``batch_id``.
        """
        import torch

        self._ensure_loaded()
        tokenizer, model = self._tokenizer, self._model

        # --- build per-item chat texts ---
        texts: list[str] = []
        for item in items:
            chat: list[dict[str, str]] = []
            if item.system is not None:
                chat.append({"role": "system", "content": item.system})
            chat.extend({"role": m.role, "content": m.content} for m in item.messages)
            texts.append(
                tokenizer.apply_chat_template(
                    chat, add_generation_prompt=True, tokenize=False, enable_thinking=False
                )
            )

        # --- batch-tokenize with left-padding (required for decoder-only models) ---
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        tokenizer.padding_side = orig_padding_side

        # --- merge gen_kwargs: use the first item's non-schema kwargs as the base ---
        # All items in a batch share the same generation hyperparams; only
        # _json_schema differs per item.
        base_kwargs = {k: v for k, v in items[0].gen_kwargs.items() if k != "_json_schema"}

        # --- build combined prefix_allowed_tokens_fn if any item has a schema ---
        schemas = [item.gen_kwargs.get("_json_schema") for item in items]
        if any(s is not None for s in schemas):
            try:
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_transformers_prefix_allowed_tokens_fn,
                )
            except ImportError as exc:
                raise ImportError(
                    "Constrained JSON decoding requires lm-format-enforcer. "
                    'Install it with: pip install -e ".[local]"'
                ) from exc

            vocab_size = tokenizer.vocab_size
            per_item_fns = [
                build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(s))
                if s is not None else None
                for s in schemas
            ]

            def combined_prefix_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
                fn = per_item_fns[batch_id]
                if fn is None:
                    return list(range(vocab_size))
                return fn(batch_id, input_ids)

            base_kwargs["prefix_allowed_tokens_fn"] = combined_prefix_fn

        t0 = time.monotonic()
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, **base_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.warning("CUDA OOM — clearing cache and retrying once")
            try:
                with torch.inference_mode():
                    generated = model.generate(**inputs, **base_kwargs)
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(
                    "CUDA out of memory after cache clear. "
                    "Try a smaller model, reduce max_new_tokens, or use device='cpu'."
                ) from exc
        log.debug("generate() batch=%d took %.2fs", len(items), time.monotonic() - t0)

        # --- slice output per item (left-padding means new tokens start at the same offset) ---
        max_prompt_len = inputs["input_ids"].shape[1]
        responses: list[LLMResponse] = []
        for i, item in enumerate(items):
            prompt_tokens = int(inputs["attention_mask"][i].sum())
            new_tokens = generated[i][max_prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(
                LLMResponse(
                    text=text,
                    model=self.model,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": int(new_tokens.shape[-1]),
                        "total_tokens": prompt_tokens + int(new_tokens.shape[-1]),
                    },
                    raw=None,
                )
            )
        return responses

    async def _drain_loop(self) -> None:
        """Background task: drain the request queue in batches."""
        while True:
            item = await self._queue.get()
            if item is None:
                break

            batch = [item]
            if self.max_batch_size > 1:
                deadline = asyncio.get_event_loop().time() + self.batch_timeout_sec
                while len(batch) < self.max_batch_size:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                        if nxt is None:
                            await self._queue.put(None)
                            break
                        batch.append(nxt)
                    except asyncio.TimeoutError:
                        break

            log.debug("Draining batch of %d request(s)", len(batch))
            try:
                responses = await asyncio.to_thread(self._run_batch, batch)
                for it, resp in zip(batch, responses):
                    if not it.future.done():
                        it.future.set_result(resp)
            except Exception as exc:
                for it in batch:
                    if not it.future.done():
                        it.future.set_exception(exc)

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

        json_schema = kwargs.pop("json_schema", None)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, **kwargs}
        if json_schema is not None:
            gen_kwargs["_json_schema"] = json_schema

        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain_loop())

        loop = asyncio.get_running_loop()
        future: asyncio.Future[LLMResponse] = loop.create_future()
        await self._queue.put(_BatchItem(messages, system, gen_kwargs, future))
        return await future
