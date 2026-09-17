"""
AI Engine Configuration.
"""
import os
from typing import Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration schema for AI Engine and pluggable models."""
    
    model_provider: str = Field(
        default="custom", 
        description="Type of provider: 'custom', 'huggingface', 'ollama', 'vllm', 'openai', 'sarvam', 'mock'"
    )
    model_name_or_path: str = Field(
        default="Qwen/Qwen3-32B",
        description="HuggingFace model ID, local directory path, or API model name"
    )
    
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1)
    context_window_size: int = Field(default=10)
    
    device: str = Field(default="cpu")
    dtype: str = Field(default="float16")
    
    api_base_url: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None)

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            model_provider=os.getenv("AI_MODEL_PROVIDER", "custom"),
            model_name_or_path=os.getenv("AI_MODEL_NAME_OR_PATH", os.getenv("MAITRI_PRIMARY_MODEL", "custom-model")),
            temperature=float(os.getenv("AI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("AI_MAX_TOKENS", "512")),
            device=os.getenv("AI_DEVICE", "cpu"),
            api_base_url=os.getenv("AI_API_BASE_URL", None),
            api_key=os.getenv("AI_API_KEY", None),
        )


default_model_config = ModelConfig.from_env()
