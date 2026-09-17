"""
Mock / Standalone Offline AI Model for Mythri.
"""
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, List
from .base_model import BaseAIModel


class MockAIModel(BaseAIModel):
    """
    Zero-dependency offline model for testing and development.
    """

    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "end my life", "want to die", 
        "hang myself", "cut myself", "self harm", "overdose"
    ]

    EMOTION_PATTERNS = {
        "Anxious": ["anxious", "panic", "worried", "nervous", "stressed", "fear", "scared", "overwhelmed"],
        "Sad": ["sad", "depressed", "unhappy", "crying", "hopeless", "lonely", "hurt", "grief"],
        "Angry": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated", "hate"],
        "Joyful": ["happy", "great", "awesome", "excited", "glad", "wonderful", "celebrate", "relieved"],
        "Fatigued": ["tired", "exhausted", "burnt out", "sleepy", "drained", "no energy"]
    }

    EMOTION_EMOJIS = {
        "Anxious": "😰",
        "Sad": "😢",
        "Angry": "😠",
        "Joyful": "😊",
        "Fatigued": "🥱",
        "Neutral": "😐"
    }

    def analyze_state(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        text_lower = text.lower()
        
        for kw in self.CRISIS_KEYWORDS:
            if kw in text_lower:
                return {
                    "emotion": "Crisis",
                    "emotion_emoji": "🚨",
                    "distress_score": 0.95,
                    "arousal_score": 0.90,
                    "primary_concern": "Crisis / Self-Harm",
                    "risk_level": "critical",
                    "is_safe": False
                }

        detected_emotion = "Neutral"
        distress = 0.2
        arousal = 0.3
        
        for emotion, keywords in self.EMOTION_PATTERNS.items():
            if any(k in text_lower for k in keywords):
                detected_emotion = emotion
                if emotion in ["Anxious", "Sad", "Angry"]:
                    distress = 0.7
                    arousal = 0.8 if emotion in ["Anxious", "Angry"] else 0.4
                elif emotion == "Joyful":
                    distress = 0.05
                    arousal = 0.7
                elif emotion == "Fatigued":
                    distress = 0.5
                    arousal = 0.1
                break

        return {
            "emotion": detected_emotion,
            "emotion_emoji": self.EMOTION_EMOJIS.get(detected_emotion, "😐"),
            "distress_score": distress,
            "arousal_score": arousal,
            "primary_concern": f"{detected_emotion} state detected" if detected_emotion != "Neutral" else "General conversation",
            "risk_level": "low" if distress < 0.6 else "moderate",
            "is_safe": True
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        analysis = self.analyze_state(prompt, context)
        
        if not analysis.get("is_safe", True):
            return (
                "I hear how much pain you're going through right now, and I want you to be safe. "
                "Please connect with someone who can support you right away:\n\n"
                "• Tele-MANAS (India): 14416 (24x7 Free)\n"
                "• Kiran Helpline: 1800-599-0019\n"
                "• Emergency: 112\n\n"
                "You are not alone, and help is available."
            )

        emotion = analysis["emotion"]
        if emotion == "Anxious":
            return (
                f"It sounds like there's a lot weighing on your mind right now. "
                f"Take a slow breath with me — what feels like the biggest piece of this at the moment?"
            )
        elif emotion == "Sad":
            return (
                f"I'm really sorry you're going through this. I'm right here with you. "
                f"Do you want to talk more about what's been happening, or just take it one thought at a time?"
            )
        elif emotion == "Angry":
            return (
                f"That sounds genuinely frustrating, and it completely makes sense why you'd feel that way. "
                f"What happened?"
            )
        elif emotion == "Joyful":
            return (
                f"That is wonderful to hear! I love seeing you in good spirits. Tell me more about it!"
            )
        elif emotion == "Fatigued":
            return (
                f"You sound really drained. Be gentle with yourself today. What's been taking up most of your energy lately?"
            )
        else:
            return (
                f"I'm listening. Tell me more about what's on your mind."
            )

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        await asyncio.sleep(0.05)
        return self.generate(prompt, system_prompt, conversation_history, context, **kwargs)

    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        full_text = self.generate(prompt, system_prompt, conversation_history, context, **kwargs)
        words = full_text.split(" ")
        for word in words:
            await asyncio.sleep(0.02)
            yield word + " "
