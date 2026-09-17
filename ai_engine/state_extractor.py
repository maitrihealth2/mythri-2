from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class UserState(BaseModel):
    """
    10 Core Parameters representing user psychological and conversational state.
    """
    emotion: str = Field(default="Neutral", description="Primary emotion detected")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    distress: float = Field(default=0.2, ge=0.0, le=1.0)
    intent: str = Field(default="Casual", description="Goal of message")
    arousal: float = Field(default=0.3, ge=0.0, le=1.0)
    sensitivity: float = Field(default=0.3, ge=0.0, le=1.0)
    engagement: float = Field(default=0.7, ge=0.0, le=1.0)
    concern: str = Field(default="General discussion")
    risk_level: str = Field(default="Low")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)


class StateExtractor:
    """
    Converts raw model outputs into structured UserState.
    """
    @staticmethod
    def extract_state(data: Dict[str, Any]) -> UserState:
        try:
            return UserState(**data)
        except Exception as e:
            return UserState(
                emotion=data.get("emotion", "Neutral"),
                distress=float(data.get("distress", data.get("distress_score", 0.2))),
                risk_level=data.get("risk_level", "Low"),
                concern=data.get("concern", data.get("primary_concern", "General"))
            )
