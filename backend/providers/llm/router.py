"""
LLM Router — Mythri's provider-agnostic inference orchestrator.

Routing flow:
    router.generate(...)
        ↓
    SarvamProvider.generate()

Failover triggers:
    None currently (only one provider is active).
"""
import time
import asyncio
from typing import Optional

from providers.llm.sarvam import SarvamProvider
from providers.llm.exceptions import (
    ProviderConfigurationError,
    ProviderEmptyResponseError,
    ProviderError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderStreamError,
    ProviderTimeoutError,
)

def _log(msg: str) -> None:
    """Structured provider telemetry — printed to stdout for log capture."""
    print(f"[LLM] {msg}", flush=True)


class LLMRouter:
    """
    Provider-agnostic LLM router.

    Singleton instance exposed as `llm_router` at module level.
    """

    def __init__(self) -> None:
        self._primary = SarvamProvider()

    async def generate(
        self,
        api_messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.75,
    ) -> Optional[str]:
        """
        Generate a response using the primary provider (Sarvam).

        Returns the response text, or None if the provider fails.
        """
        t0 = time.perf_counter()
        provider = self._primary

        try:
            result = await provider.generate(api_messages, max_tokens, temperature)
            elapsed = time.perf_counter() - t0
            _log(
                f"LLM_PROVIDER={provider.name} "
                f"LLM_MODEL={provider.model} "
                f"RESPONSE_TIME={elapsed:.2f}s "
                f"STREAMING_ENABLED=True "
                f"STREAM_COMPLETED=True"
            )
            return result

        except (ProviderConfigurationError, ProviderError) as exc:
            reason = _classify_reason(exc) if isinstance(exc, ProviderError) else "NotConfigured"
            _log(f"Delegating to pluggable ai_engine (reason: {reason})")
            try:
                from ai_engine.engine import ai_engine
                # Extract prompt and system from api_messages
                system_prompt = ""
                history = []
                user_prompt = ""
                for msg in api_messages:
                    if msg.get("role") == "system":
                        system_prompt = msg.get("content", "")
                    elif msg.get("role") == "user":
                        user_prompt = msg.get("content", "")
                    elif msg.get("role") == "assistant":
                        history.append(msg)
                
                return await ai_engine.generate_response(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    conversation_history=history,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            except Exception as ai_err:
                _log(f"ai_engine generation error: {ai_err}")
                return "I'm here with you. Please tell me more about what's on your mind."

    async def stream(
        self,
        api_messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.75,
    ):
        """
        Stream response dynamically from the primary provider or pluggable ai_engine.
        """
        provider = self._primary
        try:
            async for chunk in provider.stream(api_messages, max_tokens, temperature):
                yield chunk
        except (ProviderConfigurationError, ProviderError) as exc:
            reason = _classify_reason(exc) if isinstance(exc, ProviderError) else "NotConfigured"
            _log(f"Streaming via pluggable ai_engine (reason: {reason})")
            try:
                from ai_engine.engine import ai_engine
                system_prompt = ""
                history = []
                user_prompt = ""
                for msg in api_messages:
                    if msg.get("role") == "system":
                        system_prompt = msg.get("content", "")
                    elif msg.get("role") == "user":
                        user_prompt = msg.get("content", "")
                    elif msg.get("role") == "assistant":
                        history.append(msg)

                async for chunk in ai_engine.stream_response(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    conversation_history=history,
                    max_tokens=max_tokens,
                    temperature=temperature
                ):
                    yield chunk
            except Exception as ai_err:
                _log(f"ai_engine stream error: {ai_err}")
                yield "I'm listening, go on."
                return

    async def close(self) -> None:
        """Release all open HTTP connections."""
        if self._primary:
            await self._primary.close()


def _classify_reason(exc: ProviderError) -> str:
    if isinstance(exc, ProviderTimeoutError):
        return "Timeout"
    if isinstance(exc, ProviderRateLimitError):
        return "HTTP429"
    if isinstance(exc, ProviderServerError):
        return "HTTP5xx"
    if isinstance(exc, ProviderNetworkError):
        return "NetworkFailure"
    if isinstance(exc, ProviderStreamError):
        return "StreamingFailure"
    if isinstance(exc, ProviderEmptyResponseError):
        return "EmptyResponse"
    return "Unknown"


# ---------------------------------------------------------------------------
# Module-level singleton — import this in sarvam_client.py
# ---------------------------------------------------------------------------
llm_router = LLMRouter()

