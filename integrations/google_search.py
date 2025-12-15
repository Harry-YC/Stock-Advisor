"""
Google Search Grounding Integration

Uses Gemini's native grounding with Google Search for real-time data.
Replaces Tavily for web search functionality.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

# Model name mapping: OpenAI-compatible names → Native SDK names
# The OpenAI-compatible endpoint accepts different model names than the native SDK
MODEL_NAME_MAPPING = {
    "gemini-3-pro-preview": "gemini-2.0-flash",  # Fallback for preview models
    "gemini-3-pro": "gemini-2.0-flash",
    "gemini-3.0-pro-preview": "gemini-2.0-flash",
    "gemini-3.0-pro": "gemini-2.0-flash",
    # Add more mappings as needed
}

def _normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for native google-generativeai SDK.

    The OpenAI-compatible endpoint accepts different model names than the native SDK.
    This maps preview/internal names to production SDK names.
    """
    if model_name in MODEL_NAME_MAPPING:
        mapped = MODEL_NAME_MAPPING[model_name]
        logger.debug(f"Mapped model '{model_name}' → '{mapped}' for native SDK")
        return mapped
    return model_name

@dataclass
class GroundingSource:
    """A source from Google Search grounding."""
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}

@dataclass
class GroundedResponse:
    """Response with grounding metadata."""
    content: str
    sources: List[GroundingSource] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    grounding_chunks: List[Dict] = field(default_factory=list)  # segment→source mappings

class GoogleSearchClient:
    """
    Client for Gemini with Google Search grounding.
    
    Uses native google-generativeai SDK (not OpenAI-compatible endpoint).
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._model = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_model(self, model_name: str = "gemini-2.0-flash"):
        """Get or initialize the generative model."""
        # Normalize model name for native SDK compatibility
        normalized_model = _normalize_model_name(model_name)

        # Re-initialize if model name changed
        if self._model is None or getattr(self, '_current_model', None) != normalized_model:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(normalized_model)
                self._current_model = normalized_model
                logger.info(f"Initialized Google Generative AI model: {normalized_model}")
            except ImportError:
                logger.error("google-generativeai package not installed")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize Google Generative AI model '{normalized_model}': {e}")
                return None
        return self._model

    def generate_with_grounding(
        self,
        prompt: str,
        system_instruction: str = "",
        model_name: str = "gemini-2.0-flash",
        dynamic_threshold: float = None
    ) -> GroundedResponse:
        """
        Generate response with Google Search grounding.

        Args:
            prompt: User prompt
            system_instruction: System context
            model_name: Gemini model to use
            dynamic_threshold: 0-1, lower = more grounding (defaults to settings)

        Returns:
            GroundedResponse with content and source metadata
        """
        if not self.is_available():
            raise ValueError("Google API Key not set")

        try:
            import google.generativeai as genai
            from google.generativeai import types
        except ImportError:
            raise RuntimeError("google-generativeai package not installed")

        model = self._get_model(model_name)
        if not model:
            raise RuntimeError("Failed to initialize model")

        # Configure grounding tool - use SDK-appropriate format
        # Try different formats for SDK compatibility
        try:
            # SDK >= 0.8.x format using Tool class
            from google.generativeai.types import Tool
            tools = Tool.from_google_search_retrieval(
                google_search_retrieval=types.GoogleSearchRetrieval()
            )
        except (AttributeError, TypeError):
            try:
                # Alternative: direct GoogleSearchRetrieval
                tools = types.GoogleSearchRetrieval()
            except (AttributeError, TypeError):
                # Fallback: dictionary format for older SDKs
                tools = [{'google_search_retrieval': {}}]

        # Dynamic retrieval config
        threshold = dynamic_threshold if dynamic_threshold is not None else settings.GOOGLE_SEARCH_GROUNDING_THRESHOLD

        generation_config = types.GenerationConfig(
            temperature=0.7
        )

        try:
            response = model.generate_content(
                prompt,
                tools=tools,
                generation_config=generation_config,
                # system_instruction is supported in newer SDKs as argument to GenerativeModel or generate_content?
                # It is usually constructor arg. But let's try passing in content if supported or prepend to prompt.
                # safely let's prepend system instruction to prompt as fallback if needed or use constructor.
            )
            
            # Since we initialized model without system_instruction in _get_model singleton, 
            # we should technically re-init or prepend. 
            # Ideally, we should initialize model with system_instruction if provided.
            # Reworking _get_model to support re-init or just prepending for safety.
            
            # Actually, let's prepend to be safe and simple for this implementation
            full_prompt = f"System: {system_instruction}\n\nUser: {prompt}" if system_instruction else prompt
            
            response = model.generate_content(
                full_prompt,
                tools=tools,
                generation_config=generation_config
            )

            # Extract grounding metadata
            sources = []
            search_queries = []
            grounding_chunks = []

            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Accessing text safely
                content_text = ""
                try:
                    content_text = response.text
                except ValueError:
                    # Fallback for safety blocked responses
                     if candidate.finish_reason != 1: # STOP
                         content_text = f"[Response Blocked: {candidate.finish_reason}]"

                if hasattr(candidate, 'grounding_metadata'):
                    metadata = candidate.grounding_metadata

                    # Extract search queries used
                    if hasattr(metadata, 'search_entry_point'):
                        if hasattr(metadata.search_entry_point, 'rendered_content'):
                            search_queries = [metadata.search_entry_point.rendered_content]

                    # Extract web sources
                    if hasattr(metadata, 'grounding_chunks'):
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, 'web'):
                                sources.append(GroundingSource(
                                    title=chunk.web.title or "Web Source",
                                    url=chunk.web.uri or "",
                                    snippet="" # Snippet not always available in chunk directly
                                ))

                    # Extract segment→source mappings for inline citations
                    if hasattr(metadata, 'grounding_supports'):
                        for support in metadata.grounding_supports:
                             # Convert repeated scalar fields to lists
                            indices = []
                            if hasattr(support, 'grounding_chunk_indices'):
                                indices = list(support.grounding_chunk_indices)
                            
                            segment_text = ""
                            if hasattr(support, 'segment') and hasattr(support.segment, 'text'):
                                segment_text = support.segment.text

                            grounding_chunks.append({
                                "segment": segment_text,
                                "source_indices": indices
                            })

            return GroundedResponse(
                content=content_text,
                sources=sources,
                search_queries=search_queries,
                grounding_chunks=grounding_chunks
            )

        except Exception as e:
            logger.error(f"Grounding generation failed: {e}")
            raise e


# Convenience functions (drop-in replacements for Tavily)

def search_with_grounding(
    question: str,
    system_context: str = "",
    max_sources: int = 5
) -> Tuple[str, List[Dict]]:
    """
    Search using Google grounding and return formatted results.

    Returns:
        Tuple of (response_text, sources_list)
    """
    client = GoogleSearchClient()
    if not client.is_available():
        return "", []

    try:
        result = client.generate_with_grounding(
            prompt=question,
            system_instruction=system_context
        )
        sources = [s.to_dict() for s in result.sources[:max_sources]]
        return result.content, sources
    except Exception as e:
        logger.error(f"Search with grounding failed: {e}")
        return "", []

def format_grounding_sources(sources: List[Dict]) -> str:
    """Format sources for injection into prompts (like Tavily's format_web_results)."""
    if not sources:
        return ""

    lines = []
    for i, source in enumerate(sources, 1):
        lines.append(f"[W{i}] {source.get('title', 'Web Source')[:60]}")
        lines.append(f"    URL: {source.get('url', '')[:80]}")
        if source.get('snippet'):
            lines.append(f"    {source['snippet'][:200]}...")
        lines.append("")

    return "\n".join(lines)
