"""
Google Search & Maps Grounding Integration

Uses Gemini's native grounding with Google Search and Google Maps for real-time data.
- Google Search: General web information, reviews, articles
- Google Maps: 250M+ places, live traffic, routing, business hours, ratings
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
    places: List[Dict] = field(default_factory=list)  # Google Maps places data
    map_context_token: Optional[str] = None  # For embedding Maps widget

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


# =============================================================================
# TRAVEL-SPECIFIC GROUNDING (Search + Maps Combined)
# =============================================================================

# Destination coordinates for Maps grounding (major travel destinations)
DESTINATION_COORDINATES = {
    # New Zealand
    "auckland": (36.8509, 174.7645),
    "wellington": (-41.2924, 174.7787),
    "christchurch": (-43.5321, 172.6362),
    "queenstown": (-45.0312, 168.6626),
    # Norway
    "oslo": (59.9139, 10.7522),
    "bergen": (60.3913, 5.3221),
    "trondheim": (63.4305, 10.3951),
    "tromso": (69.6496, 18.9560),
    # Japan
    "tokyo": (35.6762, 139.6503),
    "osaka": (34.6937, 135.5023),
    "kyoto": (35.0116, 135.7681),
    "sapporo": (43.0618, 141.3545),
    # Popular destinations
    "paris": (48.8566, 2.3522),
    "london": (51.5074, -0.1278),
    "new york": (40.7128, -74.0060),
    "barcelona": (41.3874, 2.1686),
    "rome": (41.9028, 12.4964),
    "bali": (-8.3405, 115.0920),
    "bangkok": (13.7563, 100.5018),
    "singapore": (1.3521, 103.8198),
    "sydney": (-33.8688, 151.2093),
    "dubai": (25.2048, 55.2708),
}


class TravelGroundingClient:
    """
    Client for Gemini with combined Google Search + Maps grounding.

    Uses the new google.genai SDK for full grounding capabilities.
    Optimized for travel planning with location-aware queries.
    """

    # Models that support Maps grounding (NOT Gemini 3)
    MAPS_SUPPORTED_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        """Get or initialize the genai client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Google GenAI client for Maps+Search grounding")
            except ImportError:
                logger.error("google-genai package not installed. Run: pip install google-genai")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize GenAI client: {e}")
                return None
        return self._client

    def _get_coordinates(self, destination: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a destination."""
        dest_lower = destination.lower().strip()

        # Direct match
        if dest_lower in DESTINATION_COORDINATES:
            return DESTINATION_COORDINATES[dest_lower]

        # Partial match
        for key, coords in DESTINATION_COORDINATES.items():
            if key in dest_lower or dest_lower in key:
                return coords

        return None

    def generate_with_travel_grounding(
        self,
        prompt: str,
        destination: Optional[str] = None,
        include_maps: bool = True,
        include_search: bool = False,  # Default OFF - use Maps only
        system_context: str = "",
        model: str = "gemini-2.5-flash"
    ) -> GroundedResponse:
        """
        Generate response with Google Maps grounding (optionally with Search).

        Default: Maps grounding ONLY for place data (hotels, restaurants, attractions).
        This provides real ratings, reviews, hours without web search overhead.

        Args:
            prompt: User query about travel
            destination: Optional destination for location-aware Maps queries
            include_maps: Enable Maps grounding (ratings, hours, places) - default ON
            include_search: Enable Search grounding (articles, news) - default OFF
            system_context: Additional system instructions
            model: Model to use (must support Maps grounding)

        Returns:
            GroundedResponse with content, sources, and places data
        """
        if not self.is_available():
            raise ValueError("Google API Key not set")

        client = self._get_client()
        if not client:
            raise RuntimeError("Failed to initialize GenAI client")

        try:
            from google.genai import types
        except ImportError:
            raise RuntimeError("google-genai package not installed")

        # Build tools list - Maps only by default
        tools = []

        if include_maps and model in self.MAPS_SUPPORTED_MODELS:
            tools.append(types.Tool(google_maps=types.GoogleMaps()))

        # Only add Search if explicitly requested
        if include_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        if not tools:
            raise ValueError("At least one grounding tool must be enabled")

        # Build config with optional location
        config_kwargs = {"tools": tools}

        # Add location for Maps grounding if destination provided
        if include_maps and destination:
            coords = self._get_coordinates(destination)
            if coords:
                config_kwargs["tool_config"] = types.ToolConfig(
                    retrieval_config=types.RetrievalConfig(
                        lat_lng=types.LatLng(
                            latitude=coords[0],
                            longitude=coords[1]
                        )
                    )
                )
                logger.debug(f"Using coordinates {coords} for {destination}")

        # Add system instruction to prompt
        full_prompt = prompt
        if system_context:
            full_prompt = f"{system_context}\n\n{prompt}"

        try:
            response = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=types.GenerateContentConfig(**config_kwargs)
            )

            # Extract content and grounding metadata
            content_text = ""
            sources = []
            places = []
            map_token = None

            if hasattr(response, 'text'):
                content_text = response.text

            # Extract grounding metadata from candidates
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                if hasattr(candidate, 'grounding_metadata'):
                    metadata = candidate.grounding_metadata

                    # Web sources from Search grounding
                    if hasattr(metadata, 'grounding_chunks'):
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, 'web'):
                                sources.append(GroundingSource(
                                    title=getattr(chunk.web, 'title', '') or "Web Source",
                                    url=getattr(chunk.web, 'uri', '') or "",
                                    snippet=""
                                ))

                    # Places data from Maps grounding
                    if hasattr(metadata, 'grounding_supports'):
                        for support in metadata.grounding_supports:
                            if hasattr(support, 'segment'):
                                places.append({
                                    "text": getattr(support.segment, 'text', ''),
                                    "type": "maps_grounded"
                                })

                    # Maps widget token for embedding
                    if hasattr(metadata, 'google_maps_widget_context_token'):
                        map_token = metadata.google_maps_widget_context_token

            return GroundedResponse(
                content=content_text,
                sources=sources,
                search_queries=[],
                grounding_chunks=[],
                places=places,
                map_context_token=map_token
            )

        except Exception as e:
            logger.error(f"Travel grounding generation failed: {e}")
            raise e


def search_travel_info(
    question: str,
    destination: Optional[str] = None,
    include_maps: bool = True,
    include_search: bool = False,  # Default OFF - Maps only
    system_context: str = "",
    max_sources: int = 10
) -> GroundedResponse:
    """
    Search for travel information using Google Maps grounding.

    Default: Maps grounding ONLY (250M+ places with ratings, hours, reviews).
    No web search unless explicitly enabled.

    Args:
        question: Travel-related question
        destination: Optional destination city for location-aware results
        include_maps: Enable Maps grounding (default True)
        include_search: Enable web search grounding (default False)
        system_context: Additional context for the query
        max_sources: Maximum sources to return

    Returns:
        GroundedResponse with content, sources, and places data
    """
    # Try TravelGroundingClient with Maps grounding
    try:
        client = TravelGroundingClient()
        if client.is_available():
            travel_context = (
                "You are a travel planning expert. Provide specific, actionable information "
                "including real ratings, prices, hours, and practical tips. "
                "For hotels and restaurants, include star ratings and review counts. "
                "For attractions, include opening hours and ticket prices when available."
            )
            if system_context:
                travel_context = f"{travel_context}\n\n{system_context}"

            result = client.generate_with_travel_grounding(
                prompt=question,
                destination=destination,
                include_maps=include_maps,
                include_search=include_search,  # Default False
                system_context=travel_context
            )
            return result
    except Exception as e:
        logger.warning(f"TravelGroundingClient failed: {e}, returning empty response")
        return GroundedResponse(content="", sources=[], search_queries=[])
