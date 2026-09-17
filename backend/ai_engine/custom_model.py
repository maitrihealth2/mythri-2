"""
==============================================================================
MYTHRI AI ENGINE - CUSTOM MODEL INTEGRATION
==============================================================================

👉 THIS IS WHERE YOU UPDATE AND PLUG IN YOUR OWN MODEL! 👈
"""
import os
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, List
from .base_model import BaseAIModel
from .mock_model import MockAIModel
from .config import ModelConfig, default_model_config


class CustomAIModel(BaseAIModel):
    """
    Your custom AI model implementation.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or default_model_config
        self.fallback_model = MockAIModel()
        print(f"[AIEngine] Initialized CustomAIModel with config: provider={self.config.model_provider}, model={self.config.model_name_or_path}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        return self.fallback_model.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            context=context,
            **kwargs
        )

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        return await self.fallback_model.generate_async(
            prompt=prompt,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            context=context,
            **kwargs
        )

    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        async for chunk in self.fallback_model.stream_async(
            prompt=prompt,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            context=context,
            **kwargs
        ):
            yield chunk

    def analyze_state(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.fallback_model.analyze_state(text, context)
