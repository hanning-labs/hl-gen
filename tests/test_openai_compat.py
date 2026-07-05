"""Unit tests for OpenAICompatClient — no server; the OpenAI SDK client is faked."""

from __future__ import annotations

from types import SimpleNamespace

from llm import Message
from llm.openai_compat import OpenAICompatClient


class _FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _response(content, reasoning_content=None, model="test-model"):
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        model=model,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _client(response):
    client = OpenAICompatClient(model="test-model")
    fake = _FakeCompletions(response)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


async def test_system_and_message_mapping():
    client, fake = _client(_response("hi"))
    await client.complete([Message(role="user", content="hello")], system="be brief")
    call = fake.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert call["model"] == "test-model"
    assert call["max_tokens"] == client.max_new_tokens


async def test_max_new_tokens_maps_to_max_tokens():
    client, fake = _client(_response("hi"))
    await client.complete([Message(role="user", content="x")], max_new_tokens=99)
    assert fake.calls[0]["max_tokens"] == 99
    assert "max_new_tokens" not in fake.calls[0]


async def test_json_schema_maps_to_response_format():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    client, fake = _client(_response("{}"))
    await client.complete([Message(role="user", content="x")], json_schema=schema)
    fmt = fake.calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"] == schema


async def test_server_side_reasoning_content_preferred():
    client, _ = _client(_response("answer", reasoning_content="because"))
    resp = await client.complete([Message(role="user", content="x")])
    assert resp.text == "answer"
    assert resp.reasoning == "because"


async def test_vllm_reasoning_field_name():
    # vLLM 0.24 calls the field `reasoning`, not `reasoning_content`
    message = SimpleNamespace(content="answer", reasoning="because")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)], model="test-model", usage=None
    )
    client, _ = _client(response)
    resp = await client.complete([Message(role="user", content="x")])
    assert resp.text == "answer"
    assert resp.reasoning == "because"


async def test_inline_think_tag_fallback():
    client, _ = _client(_response("<think>hmm</think>\nanswer"))
    resp = await client.complete([Message(role="user", content="x")])
    assert resp.text == "answer"
    assert resp.reasoning == "hmm"


async def test_usage_normalized():
    client, _ = _client(_response("hi"))
    resp = await client.complete([Message(role="user", content="x")])
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert resp.model == "test-model"
