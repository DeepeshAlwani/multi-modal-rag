"""
OpenRouter LLM wrapper for RAGAS
Reads configuration from .env file
Uses NVIDIA Nemotron 3 Super (free) as judge - NO OpenAI required
"""

import os
import requests
from typing import Optional, Any, List, Mapping
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenRouterLLM(LLM):
    """Custom LangChain LLM for OpenRouter - No OpenAI required"""
    
    api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"))
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=500)
    
    def __init__(self, **kwargs):
        # Auto-load from .env if not provided
        if "api_key" not in kwargs or not kwargs["api_key"]:
            kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY", "")
        if "model" not in kwargs or not kwargs["model"]:
            kwargs["model"] = os.getenv("JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
        
        super().__init__(**kwargs)
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file!")
    
    @property
    def _llm_type(self) -> str:
        return "openrouter_nemotron"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Make request to OpenRouter API"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error: Unexpected response format"
                
        except Exception as e:
            return f"Error: {str(e)}"