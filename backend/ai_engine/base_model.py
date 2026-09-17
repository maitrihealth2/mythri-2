"""
Base AI Model Interface for Mythri AI Engine.
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
        pass

    @abstractmethod
    def analyze_state(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pass
