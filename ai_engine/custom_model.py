"""
==============================================================================
MYTHRI AI ENGINE - CUSTOM MODEL INTEGRATION
==============================================================================

👉 THIS IS WHERE YOU UPDATE AND PLUG IN YOUR OWN MODEL! 👈

You can connect your model using any of the following approaches:
  1. Local PyTorch / Hugging Face pipeline (e.g. Qwen, LLaMA, Mistral)
  2. Local Inference Server (Ollama, vLLM, llama.cpp, TGI)
  3. Cloud / External API (OpenAI-compatible, OpenRouter, Groq, etc.)
  4. Custom rule-based / neural pipeline

Instructions:
  - Modify the `CustomAIModel` class below.
  - Implement `generate()`, `generate_async()`, `stream_async()`, and `analyze_state()`.
  - Check the commented examples inside this file for quick copy-paste templates!
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
    
    Update the initialization and inference logic below to load your custom model weights,
    connect to your local GPU cluster, or call your private inference endpoint.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or default_model_config
        self.fallback_model = MockAIModel()
        
        # ----------------------------------------------------------------------
        # TODO: LOAD YOUR MODEL HERE
        # ----------------------------------------------------------------------
        # Example A: Hugging Face Pipeline
        # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        # self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.config.model_name_or_path, 
        #     device_map=self.config.device,
        #     torch_dtype="auto"
        # )
        
        # Example B: OpenAI / OpenRouter / vLLM client
        # from openai import OpenAI, AsyncOpenAI
        # self.client = AsyncOpenAI(
        #     api_key=self.config.api_key or os.getenv("AI_API_KEY", "EMPTY"),
        #     base_url=self.config.api_base_url or os.getenv("AI_API_BASE_URL", "http://localhost:8000/v1")
        # )
        
        print(f"[AIEngine] Initialized CustomAIModel with config: provider={self.config.model_provider}, model={self.config.model_name_or_path}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Synchronous model generation.
        
        Replace this with your custom inference call.
        """
        # --- Default: Uses clean fallback if no custom model weights are loaded ---
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
        """
        Asynchronous model generation.
        
        Example with AsyncOpenAI / vLLM / Ollama:
        -------------------------------------------------------------
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name_or_path,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[AIEngine Error] Falling back to offline model: {e}")
            return await self.fallback_model.generate_async(prompt, system_prompt, conversation_history, context)
        -------------------------------------------------------------
        """
        # By default, runs asynchronously via fallback
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
        """
        Asynchronous streaming token generation.
        
        Example streaming with OpenAI / vLLM / Ollama:
        -------------------------------------------------------------
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model_name_or_path,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"[AIEngine Stream Error] {e}")
            async for chunk in self.fallback_model.stream_async(prompt, system_prompt, conversation_history, context):
                yield chunk
        -------------------------------------------------------------
        """
        async for chunk in self.fallback_model.stream_async(
            prompt=prompt,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            context=context,
            **kwargs
        ):
            yield chunk

    def analyze_state(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze user emotion, distress score, and risk status.
        You can replace this with your fine-tuned classifier, RoBERTa emotion model, or LLM evaluation.
        """
        return self.fallback_model.analyze_state(text, context)
