"""
LLM Router Service for Travel Planner

Provides a unified interface for Gemini LLM calls.
Uses Google's native Gemini SDK for optimal performance.
"""

import logging
import time
from typing import Optional, Dict, Any, Generator
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds
RETRYABLE_ERRORS = [
    'rate limit', 'quota', 'timeout', 'connection', 'server',
    '429', '503', '502', '504', 'overloaded', 'temporarily',
    'resource_exhausted', 'unavailable', 'deadline'
]


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable."""
    error_msg = str(error).lower()
    return any(x in error_msg for x in RETRYABLE_ERRORS)


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
    Route LLM calls to Gemini models using native SDK.

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
        self._genai_configured = False

    def _ensure_configured(self):
        """Ensure Gemini SDK is configured."""
        if not self._genai_configured:
            import google.generativeai as genai
            genai.configure(api_key=self.google_api_key)
            self._genai_configured = True

    def _get_model(self, model_name: str):
        """Get a Gemini model instance."""
        self._ensure_configured()
        import google.generativeai as genai
        return genai.GenerativeModel(model_name)

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
        model_name = model or self.expert_model
        return self._call_gemini(prompt, system, model_name, temperature, max_tokens)

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
            Dict with 'type' (chunk/complete/error) and 'content'
        """
        model_name = model or self.expert_model

        for chunk in self._call_gemini_stream(prompt, system, model_name, temperature, max_tokens):
            yield chunk

    def _call_gemini(
        self,
        prompt: str,
        system: Optional[str],
        model_name: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call Gemini using native SDK with retry logic."""
        import google.generativeai as genai

        self._ensure_configured()

        # Build the model with system instruction
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system if system else None,
            generation_config=generation_config
        )

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = model.generate_content(
                    prompt,
                    request_options={"timeout": self.default_timeout}
                )

                # Extract content
                content = ""
                if response.text:
                    content = response.text
                elif response.parts:
                    content = "".join(part.text for part in response.parts if hasattr(part, 'text'))

                # Get usage metadata
                usage = {}
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = {
                        "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                        "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                        "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                    }

                return LLMResponse(
                    content=content,
                    model=model_name,
                    finish_reason="stop",
                    usage=usage,
                    raw_response=response
                )

            except Exception as e:
                last_error = e
                if is_retryable_error(e) and attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Gemini call attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Gemini call failed after {attempt + 1} attempts: {e}")
                    raise

        # Should not reach here, but just in case
        raise last_error or Exception("Unknown error in Gemini call")

    def _call_gemini_stream(
        self,
        prompt: str,
        system: Optional[str],
        model_name: str,
        temperature: float,
        max_tokens: int
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream from Gemini using native SDK with retry logic."""
        import google.generativeai as genai

        self._ensure_configured()

        # Build the model with system instruction
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system if system else None,
            generation_config=generation_config
        )

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = model.generate_content(
                    prompt,
                    stream=True,
                    request_options={"timeout": self.default_timeout}
                )

                full_content = ""
                for chunk in response:
                    if chunk.text:
                        full_content += chunk.text
                        yield {"type": "chunk", "content": chunk.text}

                yield {
                    "type": "complete",
                    "content": full_content,
                    "finish_reason": "stop"
                }
                return  # Success, exit the retry loop

            except Exception as e:
                last_error = e
                if is_retryable_error(e) and attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Gemini stream attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Gemini stream failed after {attempt + 1} attempts: {e}")
                    yield {"type": "error", "content": str(e)}
                    return

        # All retries exhausted
        yield {"type": "error", "content": str(last_error) if last_error else "Unknown error"}


# Singleton instance for convenience
_router_instance: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    """Get or create the singleton LLM Router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance


# Convenience function for simple calls
def call_llm(
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 4000,
    temperature: float = 0.7,
    timeout: float = 120.0
) -> Dict[str, Any]:
    """
    Simple function to call Gemini LLM.

    Args:
        model: Model name
        system_prompt: System instruction
        user_message: User prompt
        max_tokens: Max output tokens
        temperature: Temperature
        timeout: Timeout in seconds

    Returns:
        Dict with 'content', 'finish_reason', 'usage'
    """
    router = get_llm_router()
    router.default_timeout = int(timeout)

    response = router.call_expert(
        prompt=user_message,
        system=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return {
        'content': response.content,
        'finish_reason': response.finish_reason,
        'usage': response.usage
    }
