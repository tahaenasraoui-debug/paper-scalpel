"""
llm_engine.py
Unified LLM inference router supporting both Local AI servers (Ollama, llama.cpp)
and Cloud OpenAI-compatible APIs (Moonshot/Kimi, OpenAI, Gemini, Anthropic, DeepSeek)
with real-time token streaming and VRAM-aware context window sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generator, List, Optional, Tuple

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI


class ProviderMode(str, Enum):
    LOCAL_OLLAMA = "Local AI — Ollama (http://localhost:11434/v1)"
    LOCAL_LLAMACPP = "Local AI — llama.cpp Server (http://localhost:8080/v1)"
    CLOUD_MOONSHOT = "Cloud API — Moonshot / Kimi (https://api.moonshot.ai/v1)"
    CLOUD_OPENAI = "Cloud API — OpenAI (https://api.openai.com/v1)"
    CLOUD_GEMINI = "Cloud API — Google Gemini (OpenAI Compatible)"
    CLOUD_ANTHROPIC = "Cloud API — Anthropic Claude (OpenAI Compatible)"
    CUSTOM_ENDPOINT = "Custom OpenAI-Compatible Endpoint"


class ExecutionStrategy(str, Enum):
    AUTO = "Auto (Smart Context Routing)"
    FULL_CONTEXT = "Full-Context Pass (Single Whole-Paper Prompt)"
    SECTION_ROUTED = "Section-Routed Map-Reduce (VRAM-Safe Local Mode)"


PROVIDER_DEFAULTS: Dict[ProviderMode, Dict[str, object]] = {
    ProviderMode.LOCAL_OLLAMA: {
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5:7b-instruct",
        "is_local": True,
        "default_ctx": 16384,
    },
    ProviderMode.LOCAL_LLAMACPP: {
        "base_url": "http://localhost:8080/v1",
        "default_model": "local-model",
        "is_local": True,
        "default_ctx": 16384,
    },
    ProviderMode.CLOUD_MOONSHOT: {
        "base_url": "https://api.moonshot.ai/v1",
        "default_model": "kimi-k2-0711-preview",
        "is_local": False,
        "default_ctx": 131072,
    },
    ProviderMode.CLOUD_OPENAI: {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "is_local": False,
        "default_ctx": 128000,
    },
    ProviderMode.CLOUD_GEMINI: {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-pro",
        "is_local": False,
        "default_ctx": 200000,
    },
    ProviderMode.CLOUD_ANTHROPIC: {
        "base_url": "https://api.anthropic.com/v1/",
        "default_model": "claude-3-7-sonnet-latest",
        "is_local": False,
        "default_ctx": 200000,
    },
    ProviderMode.CUSTOM_ENDPOINT: {
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5:7b-instruct",
        "is_local": False,
        "default_ctx": 32768,
    },
}


@dataclass
class LLMConfig:
    provider_mode: ProviderMode
    base_url: str
    model_name: str
    api_key: str = "local-no-key-required"
    temperature: float = 0.15
    num_ctx: int = 16384
    max_output_tokens: int = 3072
    execution_strategy: ExecutionStrategy = ExecutionStrategy.AUTO
    strip_references: bool = True

    @property
    def is_local(self) -> bool:
        if self.provider_mode in (ProviderMode.LOCAL_OLLAMA, ProviderMode.LOCAL_LLAMACPP):
            return True
        return (
            "localhost" in self.base_url.lower()
            or "127.0.0.1" in self.base_url.lower()
            or "0.0.0.0" in self.base_url.lower()
        )


class HybridLLMEngine:
    """
    Unified OpenAI-compatible client supporting both Local AI (Ollama / llama.cpp)
    and Cloud APIs with streaming, context window enforcement, and strategy resolution.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        effective_key = (
            config.api_key.strip()
            if config.api_key and config.api_key.strip()
            else "local-no-key-required"
        )
        self.client = OpenAI(
            base_url=config.base_url.strip(),
            api_key=effective_key,
            timeout=httpx.Timeout(300.0, connect=15.0),
        )

    def resolve_execution_strategy(self, paper_tokens: int) -> ExecutionStrategy:
        """
        Determines whether to run Full-Context or Section-Routed Map-Reduce:
        - Explicit user selection overrides Auto.
        - In Auto mode: Local models OR papers exceeding 65% of configured num_ctx
          automatically route through Section-Routed Map-Reduce to prevent truncation/OOM.
        """
        if self.config.execution_strategy != ExecutionStrategy.AUTO:
            return self.config.execution_strategy

        safe_input_budget = int(self.config.num_ctx * 0.65)
        if self.config.is_local and self.config.num_ctx <= 24576:
            return ExecutionStrategy.SECTION_ROUTED
        if paper_tokens > safe_input_budget:
            return ExecutionStrategy.SECTION_ROUTED
        return ExecutionStrategy.FULL_CONTEXT

    def stream_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        """
        Streams tokens from either Local Ollama/llama.cpp or Cloud APIs.
        Automatically injects `options.num_ctx` when communicating with Ollama.
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_prompt})

        extra_body: Dict[str, object] = {}
        if self.config.provider_mode == ProviderMode.LOCAL_OLLAMA or ":11434" in self.config.base_url:
            extra_body["options"] = {
                "num_ctx": int(self.config.num_ctx),
                "temperature": float(self.config.temperature),
            }

        try:
            request_kwargs: Dict[str, object] = {
                "model": self.config.model_name.strip(),
                "messages": messages,
                "temperature": float(self.config.temperature),
                "max_tokens": int(self.config.max_output_tokens),
                "stream": True,
            }
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            response_stream = self.client.chat.completions.create(**request_kwargs)

            for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # Support DeepSeek-R1 reasoning_content if exposed by local/cloud server
                reasoning_piece = getattr(delta, "reasoning_content", None)
                if reasoning_piece:
                    yield reasoning_piece
                content_piece = getattr(delta, "content", None)
                if content_piece:
                    yield content_piece

        except APIConnectionError as exc:
            endpoint = self.config.base_url
            raise RuntimeError(
                f"Cannot connect to LLM endpoint at `{endpoint}`. "
                f"If using Local AI, verify Ollama (`ollama serve`) or llama-server is running. "
                f"Details: {exc}"
            ) from exc
        except APIStatusError as exc:
            raise RuntimeError(
                f"LLM API returned HTTP {exc.status_code}: {exc.message}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"LLM inference error: {exc}") from exc

    @staticmethod
    def check_endpoint_and_list_models(
        base_url: str, api_key: str = "local-no-key-required"
    ) -> Tuple[bool, List[str], str]:
        """
        Tests connectivity to the configured endpoint and returns available model IDs.
        Works with Ollama, llama.cpp, OpenAI, and Moonshot.
        """
        effective_key = api_key.strip() if api_key and api_key.strip() else "local-no-key-required"
        try:
            client = OpenAI(
                base_url=base_url.strip(),
                api_key=effective_key,
                timeout=httpx.Timeout(8.0, connect=4.0),
            )
            models_resp = client.models.list()
            model_ids = sorted(
                [m.id for m in getattr(models_resp, "data", []) if getattr(m, "id", None)]
            )
            return True, model_ids, f"Connected ({len(model_ids)} models available)"
        except Exception as exc:
            return False, [], f"Endpoint check failed: {exc}"
