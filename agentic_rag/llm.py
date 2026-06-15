import json
import logging
import requests
from typing import List, Dict, Any, Optional
from agentic_rag import config

logger = logging.getLogger(__name__)

class LLMResponse:
    def __init__(self, content: str, raw: Any = None):
        self.content = content
        self.raw = raw

    def __str__(self):
        return self.content

class LLMClient:
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, response_json: bool = False) -> LLMResponse:
        raise NotImplementedError

class OllamaClient(LLMClient):
    def __init__(self):
        self.host = config.OLLAMA_HOST
        self.model = config.OLLAMA_MODEL
        self.timeout = config.OLLAMA_TIMEOUT_SECONDS

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, response_json: bool = False) -> LLMResponse:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if response_json:
            payload["format"] = "json"

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"]
            return LLMResponse(content=content, raw=data)
        except Exception as e:
            logger.error(f"Ollama chat API call failed: {e}")
            raise RuntimeError(f"Ollama failure: {e}")

class GeminiClient(LLMClient):
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        # Fallback model if not configured
        self.model = "gemini-1.5-pro"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, response_json: bool = False) -> LLMResponse:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment.")
        
        # Convert OpenAI message format to Gemini content format
        contents = []
        system_instruction = None
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            else:
                gemini_role = "user" if role == "user" else "model"
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            }
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if response_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Parse answer
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError(f"No response candidates returned: {data}")
            
            content = candidates[0]["content"]["parts"][0]["text"]
            return LLMResponse(content=content, raw=data)
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise RuntimeError(f"Gemini API failure: {e}")

class OpenAIClient(LLMClient):
    def __init__(self):
        self.api_key = config.OPENAI_API_KEY
        self.model = "gpt-4o"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, response_json: bool = False) -> LLMResponse:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment.")
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return LLMResponse(content=content, raw=data)
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise RuntimeError(f"OpenAI API failure: {e}")

class MockClient(LLMClient):
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, response_json: bool = False) -> LLMResponse:
        last_message = messages[-1]["content"] if messages else ""
        mock_reply = f"Mock LLM Response to: '{last_message[:50]}...'"
        if response_json:
            # Check if this is an evaluation/judge prompt
            if "score" in last_message.lower() or "judge" in last_message.lower():
                mock_reply = json.dumps({
                    "score": 0.85,
                    "reason": "This is a mock evaluation response since no API keys are configured."
                })
            else:
                mock_reply = json.dumps({
                    "action": "answer_user",
                    "thought": "Using mock LLM mode since no API keys are available.",
                    "action_input": {
                        "synthesis": mock_reply
                    }
                })
        return LLMResponse(content=mock_reply, raw={"mock": True})

def get_llm_client() -> LLMClient:
    provider = config.LLM_PROVIDER.lower()
    if provider == "ollama":
        return OllamaClient()
    elif provider == "gemini":
        return GeminiClient()
    elif provider == "openai":
        return OpenAIClient()
    elif provider == "mock":
        return MockClient()
    else:
        logger.warning(f"Unknown LLM provider '{provider}', falling back to MockClient")
        return MockClient()
