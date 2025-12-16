"""
LLM Router Service for Travel Planner

Provides a unified interface for Gemini LLM calls.
Uses Google's Gemini API via OpenAI-compatible endpoint.
"""

import logging
from typing import Optional, Dict, Any, Generator
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from LLM calls."""
    content: str
    model: str
    finish_reason: str
    usage: Dict[str, int]
    raw_response: Optional[Any] = None


class LLMRouter:
    """
    Route LLM calls to Gemini models.

    Usage:
        router = LLMRouter()
        response = router.call_expert("Plan a trip to Barcelona", system="You are a travel expert...")
    """

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        default_timeout: int = 120
    ):
        """
        Initialize the LLM Router.

        Args:
            google_api_key: Google API key for Gemini (uses settings default if not provided)
            default_timeout: Default timeout in seconds for API calls
        """
        self.google_api_key = google_api_key or settings.GEMINI_API_KEY
        self.default_timeout = default_timeout

        # Model assignments from settings
        self.expert_model = getattr(settings, 'EXPERT_MODEL', 'gemini-3-pro-preview')

        # Lazy-loaded client
        self._gemini_client = None

    @property
    def gemini_client(self):
        """Lazy-load Gemini client (via OpenAI-compatible endpoint)."""
        if self._gemini_client is None:
            from openai import OpenAI
            gemini_base_url = getattr(settings, 'GEMINI_BASE_URL',
                                      'https://generativelanguage.googleapis.com/v1beta/openai/')
            self._gemini_client = OpenAI(
                api_key=self.google_api_key,
                base_url=gemini_base_url,
                timeout=self.default_timeout
            )
        return self._gemini_client

    def call_expert(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> LLMResponse:
        """
        Call LLM for expert discussions.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            model: Override model (uses expert_model from settings if not provided)
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and metadata
        """
        model = model or self.expert_model
        return self._call_gemini(prompt, system, model, temperature, max_tokens)

    def call_expert_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream expert response for real-time display.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            model: Override model
            temperature: Temperature
            max_tokens: Maximum tokens

        Yields:
            Dict with 'type' (chunk/complete) and 'content'
        """
        model = model or self.expert_model

        for chunk in self._call_gemini_stream(prompt, system, model, temperature, max_tokens):
            yield chunk

    def _call_gemini(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call Gemini via OpenAI-compatible endpoint."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.gemini_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=model,
                finish_reason=response.choices[0].finish_reason or "unknown",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                raw_response=response
            )

        except Exception as e:
            logger.error(f"Gemini call failed: {e}")
            raise

    def _call_gemini_stream(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream from Gemini via OpenAI-compatible endpoint."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self.gemini_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            full_content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    yield {"type": "chunk", "content": content}

            yield {
                "type": "complete",
                "content": full_content,
                "finish_reason": "stop"
            }

        except Exception as e:
            logger.error(f"Gemini stream failed: {e}")
            yield {"type": "error", "content": str(e)}


# Singleton instance for convenience
_router_instance: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    """Get or create the singleton LLM Router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance
