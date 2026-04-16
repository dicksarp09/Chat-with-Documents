import logging
import json
import re
from typing import Optional, Dict, Any, List
from groq import Groq

from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.groq_api_key
        self.model = model or "llama-3.3-70b-versatile"

        if self.api_key:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"LLM Client initialized with model: {self.model}")
        else:
            self.client = None
            logger.warning("No LLM API key provided")

    def generate(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048
    ) -> str:
        if not self.client:
            return "LLM not available - API key required"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst assistant. Provide clear, accurate responses.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error: {str(e)}"

    def generate_json(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        json_prompt = f"""{prompt}

IMPORTANT: Return ONLY valid JSON, no markdown or explanations."""

        for attempt in range(max_retries):
            try:
                response = self.generate(json_prompt, temperature=0.1)

                cleaned = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
                cleaned = re.sub(r"```\s*", "", cleaned)
                cleaned = cleaned.strip()

                start_idx = cleaned.find("{")
                end_idx = cleaned.rfind("}")

                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    return json.loads(cleaned[start_idx : end_idx + 1])

                return json.loads(cleaned)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {}

        return {}

    def is_available(self) -> bool:
        return self.client is not None


_llm_client_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance
