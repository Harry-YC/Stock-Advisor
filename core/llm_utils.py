"""
LLM Utilities

Provides a centralized factory for creating LLM clients, ensuring
correct routing for different providers (OpenAI, Google Gemini).
Also provides retry logic for transient API failures.
"""

import os
import time
import logging
from typing import Dict, Any, Tuple, Optional, List
from openai import OpenAI, APITimeoutError, RateLimitError, APIConnectionError, APIStatusError
from config import settings

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2  # seconds, exponential backoff
RETRYABLE_STATUS_CODES = {503, 429, 500, 502, 504}

def get_llm_client(api_key: str = None, model: str = None) -> OpenAI:
    """
    Factory to get the correct OpenAI-compatible client based on the model.
    
    If the model is a Gemini model, it returns a client configured for
    Google's OpenAI-compatible endpoint.
    
    Args:
        api_key: Optional specific API key (overrides settings)
        model: Model identifier (e.g., 'gemini-3-pro', 'gpt-4o')
        
    Returns:
        OpenAI client instance
    """
    model = model or settings.REASONING_MODEL
    
    # Check if this is a Gemini model requesting Google routing
    if model and model.lower().startswith("gemini"):
        google_key = api_key or settings.GOOGLE_API_KEY
        base_url = settings.GEMINI_BASE_URL
        
        if not google_key:
            logger.warning("Gemini model requested but GOOGLE_API_KEY not set. Falling back to provided key or OpenAI key.")
            google_key = api_key or settings.OPENAI_API_KEY

        logger.debug(f"Creating Gemini adapter client for model: {model}")
        return OpenAI(
            api_key=google_key,
            base_url=base_url,
            timeout=settings.OPENAI_TIMEOUT
        )
    
    # Default to standard OpenAI
    logger.debug(f"Creating standard OpenAI client for model: {model}")
    return OpenAI(
        api_key=api_key or settings.OPENAI_API_KEY,
        timeout=settings.OPENAI_TIMEOUT
    )

def generate_with_optional_grounding(
    prompt: str,
    system_instruction: str = "",
    model: str = None,
    enable_grounding: bool = False,
    api_key: str = None
) -> Tuple[str, Optional[List[Dict]]]:
    """
    Generate content with optional Google Search grounding.

    For grounded requests: Uses native google-generativeai SDK
    For non-grounded requests: Uses existing OpenAI-compatible endpoint

    Returns:
        Tuple of (response_text, grounding_sources or None)
    """
    model = model or settings.REASONING_MODEL

    if enable_grounding and model.lower().startswith("gemini"):
        # Lazy import to avoid hard dependency if not used
        try:
            from integrations.google_search import GoogleSearchClient
            client = GoogleSearchClient(api_key=api_key)
            if client.is_available():
                result = client.generate_with_grounding(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    model_name=model
                )
                sources = [s.to_dict() for s in result.sources]
                return result.content, sources
            else:
                logger.warning("Grounding requested but GoogleSearchClient unavailable. Falling back to standard generation.")
        except Exception as e:
            logger.error(f"Grounding failed: {e}. Falling back to standard generation.")

    # Fallback to existing OpenAI-compatible path
    client = get_llm_client(api_key=api_key, model=model)
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # max_completion_tokens not supported by all OpenAI-compatible endpoints,
            # but usually ignore or supported. Let's use max_tokens if needed or default.
            # Using standard parameter name for broader compatibility
        )
        return response.choices[0].message.content, None
    except Exception as e:
        logger.error(f"Standard generation failed: {e}")
        raise e
