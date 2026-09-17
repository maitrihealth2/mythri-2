"""
Base AI Model Interface for Mythri AI Engine.

Any new model (HuggingFace, PyTorch, vLLM, Ollama, OpenAI-compatible API, custom weights)
should inherit from `BaseAIModel` and implement the core generation & analysis methods.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional, List


class BaseAIModel(ABC):
    """
    Abstract Base Class for all AI models in Mythri.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Synchronous / standard generation.
        
        Args:
            prompt: User message text.
            system_prompt: System instructions / companion persona.
            conversation_history: List of {"role": "user"|"assistant", "content": "..."} turns.
            context: Additional context (user profile, emotional state, memory).
            
        Returns:
            The generated response string.
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Asynchronous generation method.
        """
        pass

    @abstractmethod
    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronous streaming generation method yielding text chunks.
        """
        pass

    @abstractmethod
    def analyze_state(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts emotional state, distress score, and risk from the user text.
        
        Returns a dictionary with keys:
            - emotion: str (e.g. "Anxious", "Sad", "Neutral", "Joyful")
            - emotion_emoji: str (e.g. "😰", "😢", "😐", "😊")
            - distress_score: float (0.0 to 1.0)
            - arousal_score: float (0.0 to 1.0)
            - primary_concern: str
            - risk_level: str ("low", "moderate", "high", "critical")
        """
        pass
