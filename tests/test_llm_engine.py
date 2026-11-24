from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_engine import (
    ExecutionStrategy,
    HybridLLMEngine,
    LLMConfig,
    ProviderMode,
    StreamChunk,
)


def _make_chunk(content: str | None = None, reasoning_content: str | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_llm_config_is_local_and_strategy_resolution() -> None:
    local_cfg = LLMConfig(
        provider_mode=ProviderMode.LOCAL_OLLAMA,
        base_url="http://localhost:11434/v1",
        model_name="qwen2.5:7b-instruct",
        num_ctx=16384,
    )
    assert local_cfg.is_local is True
    with patch("llm_engine.OpenAI"):
        engine = HybridLLMEngine(local_cfg)
    assert engine.resolve_execution_strategy(paper_tokens=2000) == ExecutionStrategy.SECTION_ROUTED

    cloud_cfg = LLMConfig(
        provider_mode=ProviderMode.CLOUD_OPENAI,
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o",
        api_key="sk-test",
        num_ctx=128000,
    )
    assert cloud_cfg.is_local is False
    with patch("llm_engine.OpenAI"):
        cloud_engine = HybridLLMEngine(cloud_cfg)
    assert cloud_engine.resolve_execution_strategy(paper_tokens=15000) == ExecutionStrategy.FULL_CONTEXT
    assert cloud_engine.resolve_execution_strategy(paper_tokens=100000) == ExecutionStrategy.SECTION_ROUTED

    explicit_cfg = LLMConfig(
        provider_mode=ProviderMode.LOCAL_OLLAMA,
        base_url="http://localhost:11434/v1",
        model_name="qwen2.5:7b-instruct",
        execution_strategy=ExecutionStrategy.FULL_CONTEXT,
    )
    with patch("llm_engine.OpenAI"):
        explicit_engine = HybridLLMEngine(explicit_cfg)
    assert explicit_engine.resolve_execution_strategy(paper_tokens=50000) == ExecutionStrategy.FULL_CONTEXT


def test_deepseek_r1_reasoning_content_separated_from_final_output() -> None:
    cfg = LLMConfig(
        provider_mode=ProviderMode.LOCAL_OLLAMA,
        base_url="http://localhost:11434/v1",
        model_name="deepseek-r1:8b",
        num_ctx=16384,
        max_output_tokens=2048,
    )
    mock_chunks = [
        _make_chunk(reasoning_content="Let me inspect Eq. 3 first. "),
        _make_chunk(reasoning_content="Now check Table 2 deltas."),
        _make_chunk(content="### 1. Problem Statement\n"),
        _make_chunk(content="Memory bandwidth limits decoding [Sec. 3.1]."),
    ]

    with patch("llm_engine.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        mock_openai.return_value = mock_client

        engine = HybridLLMEngine(cfg)
        reasoning_buf: list[str] = []
        output_pieces = list(
            engine.stream_completion(
                system_prompt="sys",
                user_prompt="usr",
                reasoning_buffer=reasoning_buf,
            )
        )

        final_report = "".join(output_pieces)
        assert "Let me inspect Eq. 3" not in final_report
        assert "Table 2 deltas" not in final_report
        assert final_report == "### 1. Problem Statement\nMemory bandwidth limits decoding [Sec. 3.1]."
        assert engine.last_reasoning_content == "Let me inspect Eq. 3 first. Now check Table 2 deltas."
        assert "".join(reasoning_buf) == engine.last_reasoning_content

        # Verify Ollama extra_body num_ctx injection
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["options"]["num_ctx"] == 16384
        assert call_kwargs["max_tokens"] == 2048


def test_stream_completion_events_yields_tagged_channels_and_max_tokens_override() -> None:
    cfg = LLMConfig(
        provider_mode=ProviderMode.CLOUD_OPENAI,
        base_url="https://api.openai.com/v1",
        model_name="o3-mini",
        api_key="sk-test",
        max_output_tokens=4096,
    )
    mock_chunks = [
        _make_chunk(reasoning_content="step 1"),
        _make_chunk(content="Final answer"),
    ]

    with patch("llm_engine.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        mock_openai.return_value = mock_client

        engine = HybridLLMEngine(cfg)
        events = list(
            engine.stream_completion_events(
                system_prompt="sys",
                user_prompt="usr",
                max_tokens=1024,
            )
        )

        assert events == [
            StreamChunk(kind="reasoning", text="step 1"),
            StreamChunk(kind="content", text="Final answer"),
        ]
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in call_kwargs
        assert call_kwargs["max_tokens"] == 1024


def test_stream_completion_resets_reasoning_between_calls() -> None:
    cfg = LLMConfig(
        provider_mode=ProviderMode.CLOUD_OPENAI,
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o",
        api_key="sk-test",
    )

    with patch("llm_engine.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            iter([_make_chunk(reasoning_content="old thought"), _make_chunk(content="first")]),
            iter([_make_chunk(content="second")]),
        ]
        mock_openai.return_value = mock_client

        engine = HybridLLMEngine(cfg)
        assert "".join(engine.stream_completion("s", "u1")) == "first"
        assert engine.last_reasoning_content == "old thought"

        assert "".join(engine.stream_completion("s", "u2")) == "second"
        assert engine.last_reasoning_content == ""


def test_stream_completion_wraps_errors_in_runtime_error() -> None:
    cfg = LLMConfig(
        provider_mode=ProviderMode.LOCAL_OLLAMA,
        base_url="http://localhost:11434/v1",
        model_name="qwen2.5:7b-instruct",
    )
    with patch("llm_engine.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ValueError("boom")
        mock_openai.return_value = mock_client

        engine = HybridLLMEngine(cfg)
        with pytest.raises(RuntimeError, match="LLM inference error: boom"):
            list(engine.stream_completion("sys", "usr"))


def test_check_endpoint_and_list_models() -> None:
    with patch("llm_engine.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="z-model"), SimpleNamespace(id="a-model")]
        )
        mock_openai.return_value = mock_client

        ok, models, msg = HybridLLMEngine.check_endpoint_and_list_models("http://localhost:11434/v1")
        assert ok is True
        assert models == ["a-model", "z-model"]
        assert "2 models" in msg

    with patch("llm_engine.OpenAI", side_effect=OSError("connection refused")):
        ok, models, msg = HybridLLMEngine.check_endpoint_and_list_models("http://localhost:11434/v1")
        assert ok is False
        assert models == []
        assert "Endpoint check failed" in msg
