# 🧠 Mythri AI Engine - Model Customization Guide

Welcome to the **Mythri AI Engine**! This directory is your dedicated hub to configure, load, and swap your custom AI models.

---

## 📁 Directory Overview

```
ai_engine/
├── base_model.py         # Base abstract class (BaseAIModel)
├── custom_model.py       # ⭐️ YOUR MODEL FILE (Edit this to load your model!)
├── mock_model.py         # Offline, zero-dependency fallback model
├── engine.py             # Main orchestrator (AIEngine & singleton ai_engine)
├── config.py             # Model hyperparameters and device configuration
├── baseline_engine.py    # User baseline & emotional deviation tracker
├── concern_tracker.py    # Clinical concern progression tracker
├── ranking_algorithm.py  # Concern prioritization algorithms
├── state_extractor.py    # 10-parameter user state validator
└── README.md             # This guide
```

---

## 🚀 How to Update Your Model in 3 Easy Steps

### Step 1: Open `ai_engine/custom_model.py`
All model inference logic is encapsulated inside `CustomAIModel`.

### Step 2: Choose Your Model Backend & Implement

#### Option A: Hugging Face / PyTorch Local Checkpoint
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class CustomAIModel(BaseAIModel):
    def __init__(self, config=None):
        self.config = config or default_model_config
        self.tokenizer = AutoTokenizer.from_pretrained("path/to/your/weights")
        self.pipe = pipeline(
            "text-generation",
            model="path/to/your/weights",
            tokenizer=self.tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        res = self.pipe(prompt, max_new_tokens=256)
        return res[0]["generated_text"]
```

#### Option B: Local Server (Ollama / vLLM / llama.cpp)
```python
from openai import AsyncOpenAI

class CustomAIModel(BaseAIModel):
    def __init__(self, config=None):
        self.config = config or default_model_config
        # Connect to Ollama (http://localhost:11434/v1) or vLLM (http://localhost:8000/v1)
        self.client = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    async def generate_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model="your-model-name",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
```

#### Option C: Custom API Endpoint (OpenRouter, Groq, Together, etc.)
```python
from openai import AsyncOpenAI
import os

class CustomAIModel(BaseAIModel):
    def __init__(self, config=None):
        self.config = config or default_model_config
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("MAITRI_OPENROUTER_API_KEY", "your_key")
        )
```

---

## ⚙️ Environment Variables (Optional)

You can customize inference via `.env`:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `AI_MODEL_PROVIDER` | `custom`, `huggingface`, `ollama`, `vllm`, `mock` | `custom` |
| `AI_MODEL_NAME_OR_PATH` | Path to weights or model tag | `Qwen/Qwen3-32B` |
| `AI_DEVICE` | `cuda`, `cpu`, `mps` | `cpu` |
| `AI_TEMPERATURE` | Generation sampling temperature | `0.7` |
| `AI_MAX_TOKENS` | Max new tokens to generate | `512` |
| `AI_API_BASE_URL` | Endpoint URL if using remote/local server | `None` |
| `AI_API_KEY` | API Key if required | `None` |

---

## 🧪 Testing Your Model Standalone

Run a quick test directly with python:
```bash
python -c "from ai_engine import ai_engine; import asyncio; print(asyncio.run(ai_engine.generate_response('Hello! How are you?')))"
```
