"""Backend-layer tests: Ollama translation/parsing, loopback guard, factory.

All offline — the one network boundary (``OllamaClient._post``) is stubbed, so
these run with no Ollama daemon and no API key.
"""

from __future__ import annotations

import pytest

from common.config import load_settings
from common.llm_client import (
    AnthropicClient,
    LLMError,
    OllamaClient,
    ToolResult,
    _to_ollama_tools,
    make_client,
)

# --- loopback guard ----------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    ["http://localhost:11434", "http://127.0.0.1:11434", "https://localhost:1234"],
)
def test_loopback_urls_accepted(url):
    client = OllamaClient(base_url=url, model="qwen3:8b")
    assert client.base_url == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "http://evil.example.com:11434",
        "http://10.0.0.5:11434",
        "http://host.docker.internal:11434",  # not allowlisted -> still refused
        "ftp://localhost:11434",
    ],
)
def test_non_loopback_urls_rejected(url):
    with pytest.raises(LLMError):
        OllamaClient(base_url=url, model="qwen3:8b")


# --- container endpoint (host.docker.internal) -------------------------------

CONTAINER_ALLOW = ("api.anthropic.com", "host.docker.internal:11434")


def test_container_endpoint_accepted_when_allowlisted():
    client = OllamaClient(
        base_url="http://host.docker.internal:11434",
        model="qwen3:8b",
        egress_allowlist=CONTAINER_ALLOW,
    )
    assert client.base_url == "http://host.docker.internal:11434"


@pytest.mark.parametrize(
    "url",
    [
        "http://host.docker.internal:11435",  # allowlist is per-port
        "http://evil.example.com:11434",  # allowlist is per-host
    ],
)
def test_container_allowlist_does_not_widen(url):
    with pytest.raises(LLMError):
        OllamaClient(base_url=url, model="qwen3:8b", egress_allowlist=CONTAINER_ALLOW)


def test_loopback_inside_container_is_a_loud_error(monkeypatch):
    # Inside the container, loopback is the container. Fail with a
    # misconfiguration message instead of a confusing connection error.
    monkeypatch.setattr("common.llm_client._in_container", lambda: True)
    with pytest.raises(LLMError, match="inside the container"):
        OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")


# --- tool schema conversion --------------------------------------------------

def test_tool_conversion_to_ollama_shape():
    neutral = [
        {
            "name": "write_file",
            "description": "Write a file",
            "strict": True,
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]
    converted = _to_ollama_tools(neutral)
    assert converted[0]["type"] == "function"
    fn = converted[0]["function"]
    assert fn["name"] == "write_file"
    assert fn["parameters"] == neutral[0]["input_schema"]
    assert "strict" not in fn  # no Ollama equivalent; dropped


# --- message construction ----------------------------------------------------

def test_ollama_tool_results_are_one_message_per_result():
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")
    results = [
        ToolResult(tool_use_id="a", name="read_file", content="x", is_error=False),
        ToolResult(tool_use_id="b", name="write_file", content="y", is_error=True),
    ]
    msgs = client.build_tool_results(results)
    assert len(msgs) == 2
    assert msgs[0] == {"role": "tool", "content": "x", "tool_name": "read_file"}
    assert msgs[1]["tool_name"] == "write_file"


# --- response parsing (stubbed transport) ------------------------------------

def _ollama(monkeypatch, payload):
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")
    monkeypatch.setattr(client, "_post", lambda path, body: payload)
    return client


def test_complete_parses_native_tool_call(monkeypatch):
    payload = {
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "let me use the tool",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "write_file", "arguments": {"path": "p.txt", "content": "c"}}}
            ],
        },
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 20,
    }
    client = _ollama(monkeypatch, payload)
    resp = client.complete(
        [{"role": "user", "content": "hi"}], system="sys", tools=[], max_tokens=100
    )
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "write_file"
    assert call.arguments == {"path": "p.txt", "content": "c"}
    assert resp.usage == {"input_tokens": 100, "output_tokens": 20}
    # thinking is captured but kept out of the scannable text
    assert resp.thinking == "let me use the tool"
    assert resp.text == ""
    # native assistant message carries tool_calls for history replay
    assert resp.assistant_message["tool_calls"]


def test_complete_parses_plain_text(monkeypatch):
    payload = {
        "message": {"role": "assistant", "content": "Here is the summary."},
        "done_reason": "stop",
        "prompt_eval_count": 50,
        "eval_count": 8,
    }
    client = _ollama(monkeypatch, payload)
    resp = client.complete([{"role": "user", "content": "hi"}], system="sys", tools=[])
    assert resp.stop_reason == "end_turn"
    assert resp.tool_calls == []
    assert resp.text == "Here is the summary."
    assert resp.refusal is None  # Ollama never sets a refusal


def test_complete_maps_length_to_max_tokens(monkeypatch):
    payload = {"message": {"content": "truncated"}, "done_reason": "length"}
    client = _ollama(monkeypatch, payload)
    resp = client.complete([{"role": "user", "content": "hi"}], system="sys")
    assert resp.stop_reason == "max_tokens"


def test_tool_arguments_as_json_string_are_parsed(monkeypatch):
    payload = {
        "message": {
            "content": "",
            "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}],
        },
        "done_reason": "stop",
    }
    client = _ollama(monkeypatch, payload)
    resp = client.complete([{"role": "user", "content": "x"}], system="s", tools=[])
    assert resp.tool_calls[0].arguments == {"path": "a.txt"}
    assert resp.tool_calls[0].id  # synthesised when Ollama omits one


def test_complete_json_parses_object(monkeypatch):
    payload = {"message": {"content": '{"success": true, "confidence": 0.9}'}}
    client = _ollama(monkeypatch, payload)
    out = client.complete_json(system="s", user="u", schema={"type": "object"})
    assert out == {"success": True, "confidence": 0.9}


def test_complete_json_rejects_non_object(monkeypatch):
    payload = {"message": {"content": "[1, 2, 3]"}}
    client = _ollama(monkeypatch, payload)
    with pytest.raises(LLMError):
        client.complete_json(system="s", user="u", schema={"type": "object"})


# --- factory -----------------------------------------------------------------

def test_make_client_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = load_settings(load_dotenv=False)
    assert settings.llm_backend == "ollama"
    client = make_client(settings)
    assert isinstance(client, OllamaClient)
    assert settings.target_model == "qwen3:8b"  # role default follows backend


def test_make_client_anthropic_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = load_settings(load_dotenv=False)
    assert settings.target_model == "claude-opus-5"
    with pytest.raises(RuntimeError):
        make_client(settings)


def test_make_client_anthropic_with_key(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    settings = load_settings(load_dotenv=False)
    client = make_client(settings)
    assert isinstance(client, AnthropicClient)


def test_unknown_backend_rejected(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gpt5")
    with pytest.raises(ValueError):
        load_settings(load_dotenv=False)
