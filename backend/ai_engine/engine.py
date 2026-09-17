"""
Mythri AI Engine - Main Orchestration Layer.
"""
from typing import AsyncGenerator, Dict, Any, Optional, List
from .custom_model import CustomAIModel
from .config import ModelConfig, default_model_config
from .baseline_engine import BaselineEngine
from .concern_tracker import ConcernTracker
from .ranking_algorithm import RankingAlgorithm
from .state_extractor import StateExtractor


class AIEngine:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or default_model_config
        self.model = CustomAIModel(self.config)
        self.baseline_engine = BaselineEngine()
        self.concern_tracker = ConcernTracker()
        self.ranking_algorithm = RankingAlgorithm()
        self.state_extractor = StateExtractor()

    def update_model(self, custom_model_instance: Any):
        self.model = custom_model_instance
        print(f"[AIEngine] Model updated to: {type(custom_model_instance).__name__}")

    def analyze_user_input(self, user_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.model.analyze_state(user_text, context)

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        return await self.model.generate_async(
            prompt=prompt,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            context=context,
            **kwargs
        )

    async def stream_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        async for chunk in self.model.stream_async(
            prompt=prompt,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            context=context,
            **kwargs
        ):
            yield chunk


ai_engine = AIEngine()
