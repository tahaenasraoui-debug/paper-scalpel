from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

import app
from analyzer import AnalysisModule
from llm_engine import PROVIDER_DEFAULTS, ExecutionStrategy, ProviderMode
from parser import ParsedPaper


def test_sanitize_export_filename_handles_slashes_colons_and_edge_cases() -> None:
    assert (
        app.sanitize_export_filename("Paper Scalpel: O(N)/O(1) Attention? A Survey", "md")
        == "Paper_Scalpel_O_N_O_1_Attention_A_Survey_teardown.md"
    )
    assert (
        app.sanitize_export_filename("../../etc/passwd:evil*name|<>", ".json")
        == "etc_passwd_evil_name_teardown.json"
    )
    assert app.sanitize_export_filename("///:::*?\"<>|", "md") == "paper_teardown.md"
    assert app.sanitize_export_filename("", "json") == "paper_teardown.json"

    long_name = app.sanitize_export_filename("A" * 100, "md")
    assert long_name == f"{'A' * 45}_teardown.md"
    assert "/" not in long_name
    assert ":" not in long_name


def test_provider_env_overrides_do_not_break_explicit_provider_switches() -> None:
    env = {
        "LLM_BASE_URL": "http://custom-ollama:11434/v1",
        "LLM_MODEL_NAME": "deepseek-r1:8b",
        "MOONSHOT_API_KEY": "sk-moonshot",
        "OPENAI_API_KEY": "sk-openai",
        "GEMINI_API_KEY": "sk-gemini",
        "ANTHROPIC_API_KEY": "sk-ant-123",
    }
    state: dict[str, object] = {
        "last_provider": None,
        "endpoint_models": ["stale-model"],
        "endpoint_status": "Connected",
        "endpoint_base_url": "http://old",
    }

    # Initial load uses LLM_BASE_URL / LLM_MODEL_NAME defaults
    changed = app.sync_provider_state(state, ProviderMode.LOCAL_OLLAMA, env=env)
    assert changed is True
    assert state["sidebar_base_url"] == "http://custom-ollama:11434/v1"
    assert state["sidebar_model_name"] == "deepseek-r1:8b"
    assert state["endpoint_models"] == []
    assert state["endpoint_status"] == ""

    # Populate detected models, then switch explicitly to Cloud Anthropic
    state["endpoint_models"] = ["deepseek-r1:8b", "qwen2.5:7b-instruct"]
    state["endpoint_status"] = "Connected (2 models available)"

    changed = app.sync_provider_state(state, ProviderMode.CLOUD_ANTHROPIC, env=env)
    assert changed is True
    assert state["sidebar_base_url"] == PROVIDER_DEFAULTS[ProviderMode.CLOUD_ANTHROPIC]["base_url"]
    assert (
        state["sidebar_model_name"]
        == PROVIDER_DEFAULTS[ProviderMode.CLOUD_ANTHROPIC]["default_model"]
    )
    assert state["sidebar_api_key"] == "sk-ant-123"
    assert state["endpoint_models"] == []
    assert state["endpoint_status"] == ""

    # Switch explicitly to Cloud OpenAI -> must NOT inherit LLM_BASE_URL/LLM_MODEL_NAME or Moonshot key
    changed = app.sync_provider_state(state, ProviderMode.CLOUD_OPENAI, env=env)
    assert changed is True
    assert state["sidebar_base_url"] == "https://api.openai.com/v1"
    assert state["sidebar_model_name"] == "gpt-4o"
    assert state["sidebar_api_key"] == "sk-openai"
    assert state["sidebar_num_ctx"] in app.CTX_OPTIONS

    # Switch to Custom Endpoint -> env overrides are respected again
    changed = app.sync_provider_state(state, ProviderMode.CUSTOM_ENDPOINT, env=env)
    assert changed is True
    assert state["sidebar_base_url"] == "http://custom-ollama:11434/v1"
    assert state["sidebar_model_name"] == "deepseek-r1:8b"

    # Re-syncing same provider does not wipe user state
    state["endpoint_models"] = ["my-model"]
    assert app.sync_provider_state(state, ProviderMode.CUSTOM_ENDPOINT, env=env) is False
    assert state["endpoint_models"] == ["my-model"]


def test_all_provider_default_ctx_values_supported_by_slider_options() -> None:
    for provider, defaults in PROVIDER_DEFAULTS.items():
        assert int(defaults["default_ctx"]) in app.CTX_OPTIONS, f"Missing ctx for {provider}"


def test_render_export_buttons_enables_immediately_when_results_populated() -> None:
    paper = ParsedPaper(
        title="Fast/Sparse: A Benchmark",
        authors_or_header="",
        page_count=5,
        raw_markdown="raw",
        cleaned_markdown="clean",
        sections={"Abstract": "Short abstract."},
    )
    md_slot = MagicMock()
    json_slot = MagicMock()

    # Before teardown: disabled
    app.render_export_buttons(
        md_slot=md_slot,
        json_slot=json_slot,
        paper=paper,
        results={},
        model_name="qwen2.5:7b-instruct",
        strategy_name=ExecutionStrategy.SECTION_ROUTED.value,
    )
    assert md_slot.download_button.call_args.kwargs["disabled"] is True
    assert json_slot.download_button.call_args.kwargs["disabled"] is True
    assert md_slot.download_button.call_args.kwargs["file_name"] == "Fast_Sparse_A_Benchmark_teardown.md"

    # Immediately after teardown populates results: enabled with fresh report payload
    results = {AnalysisModule.EXECUTIVE_SUMMARY.value: "Verified delta [Sec. 1]."}
    app.render_export_buttons(
        md_slot=md_slot,
        json_slot=json_slot,
        paper=paper,
        results=results,
        model_name="qwen2.5:7b-instruct",
        strategy_name=ExecutionStrategy.SECTION_ROUTED.value,
    )
    md_kwargs = md_slot.download_button.call_args.kwargs
    json_kwargs = json_slot.download_button.call_args.kwargs
    assert md_kwargs["disabled"] is False
    assert json_kwargs["disabled"] is False
    assert b"Verified delta [Sec. 1]." in md_kwargs["data"]
    assert b"Verified delta [Sec. 1]." in json_kwargs["data"]


def test_streamlit_sidebar_provider_switching_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://custom-host:11434/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "custom-local-model")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-e2e")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-e2e")

    at = AppTest.from_file("app.py", default_timeout=20).run()
    assert not at.exception
    assert at.session_state["sidebar_base_url"] == "http://custom-host:11434/v1"
    assert at.session_state["sidebar_model_name"] == "custom-local-model"

    at.session_state["endpoint_models"] = ["stale-local-model"]
    at.session_state["endpoint_status"] = "Connected"

    at.sidebar.radio[0].set_value("Cloud API (Kimi / OpenAI / Gemini / Claude)").run()
    assert not at.exception
    assert at.session_state["sidebar_base_url"] == "https://api.moonshot.ai/v1"
    assert at.session_state["sidebar_model_name"] == "kimi-k2-0711-preview"
    assert at.session_state["endpoint_models"] == []

    at.sidebar.selectbox[0].set_value(ProviderMode.CLOUD_OPENAI).run()
    assert not at.exception
    assert at.session_state["sidebar_base_url"] == "https://api.openai.com/v1"
    assert at.session_state["sidebar_model_name"] == "gpt-4o"
    assert at.session_state["sidebar_api_key"] == "sk-openai-e2e"
    assert at.session_state["sidebar_num_ctx"] == 128000

    at.sidebar.selectbox[0].set_value(ProviderMode.CLOUD_ANTHROPIC).run()
    assert not at.exception
    assert at.session_state["sidebar_base_url"] == "https://api.anthropic.com/v1/"
    assert at.session_state["sidebar_model_name"] == "claude-3-7-sonnet-latest"
    assert at.session_state["sidebar_api_key"] == "sk-ant-e2e"
