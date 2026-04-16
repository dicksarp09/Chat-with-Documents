import logging
import os
import json
import re
from typing import Optional, Dict, Any
from groq import Groq

from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroqClient:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY")
        self.model = model or settings.groq_model

        if not self.api_key:
            logger.warning("No Groq API key provided. LLM features will be limited.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Initialized Groq client with model: {self.model}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self.client:
            return self._fallback_response(prompt)

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._fallback_response(prompt)

    def generate_json(
        self, prompt: str, system_prompt: Optional[str] = None, max_retries: int = 3
    ) -> Dict[str, Any]:
        json_system = (
            system_prompt
            or "You must respond with valid JSON only. No markdown, no explanations, just JSON."
        )

        for attempt in range(max_retries):
            try:
                response = self.generate(
                    prompt, temperature=0.1, system_prompt=json_system
                )

                cleaned = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
                cleaned = re.sub(r"```\s*", "", cleaned)
                cleaned = cleaned.strip()

                # Try to extract just the JSON object
                # Find the first { and last }
                start_idx = cleaned.find("{")
                end_idx = cleaned.rfind("}")

                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = cleaned[start_idx : end_idx + 1]
                    return json.loads(json_str)
                else:
                    return json.loads(cleaned)

            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parsing failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    return {}

        return {}

        return {}

        return {}

        return {}

    def generate_streaming(
        self, prompt: str, temperature: float = 0.1, system_prompt: Optional[str] = None
    ):
        if not self.client:
            yield self._fallback_response(prompt)
            return

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            yield self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        logger.warning("Using fallback response - Groq client not initialized")

        if len(prompt) > 500:
            return "Groq API key not configured. Please set GROQ_API_KEY environment variable."
        return f"Processed: {prompt[:100]}..."

    def is_available(self) -> bool:
        return self.client is not None

    def get_model_name(self) -> str:
        return self.model


_groq_client_instance: Optional[GroqClient] = None


def get_groq_client() -> GroqClient:
    global _groq_client_instance
    if _groq_client_instance is None:
        _groq_client_instance = GroqClient()
    return _groq_client_instance
