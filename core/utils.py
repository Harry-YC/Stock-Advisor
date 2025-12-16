"""
Core Utilities Module

Contains shared helper functions used across the application.
"""

import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    'extract_simple_query',
    'user_friendly_error',
]


# =============================================================================
# ERROR HANDLING
# =============================================================================

# Mapping of exception types to user-friendly messages
ERROR_MESSAGES = {
    # Network/API errors
    'ConnectionError': "Unable to connect to the service. Please check your internet connection.",
    'TimeoutError': "The request took too long. Please try again.",
    'HTTPError': "The service returned an error. Please try again later.",
    'RateLimitError': "Too many requests. Please wait a moment and try again.",
    'AuthenticationError': "Authentication failed. Please check your API key.",
    'APIError': "The AI service encountered an issue. Please try again.",

    # File processing errors
    'FileNotFoundError': "The file could not be found.",
    'PermissionError': "Permission denied. Cannot access the file.",
    'UnicodeDecodeError': "Unable to read the file. It may be in an unsupported format.",
    'pdf': "Unable to process PDF. The file may be corrupted or password-protected.",

    # Data processing errors
    'JSONDecodeError': "Unable to parse the response. Please try again.",
    'KeyError': "Missing expected data in response.",
    'ValueError': "Invalid data format received.",

    # Generic fallbacks
    'default': "An unexpected error occurred. Please try again.",
}


def user_friendly_error(e: Exception, context: str = "") -> str:
    """
    Convert an exception to a user-friendly error message.

    Logs the full exception for debugging but returns a clean message for users.

    Args:
        e: The exception that occurred
        context: Optional context about what operation failed (e.g., "processing document")

    Returns:
        A user-friendly error message string
    """
    # Log the full exception for debugging
    logger.error(f"Error in {context}: {type(e).__name__}: {str(e)}", exc_info=True)

    # Get the exception type name
    exc_type = type(e).__name__
    exc_str = str(e).lower()

    # Try to match specific error types
    if exc_type in ERROR_MESSAGES:
        message = ERROR_MESSAGES[exc_type]
    # Check for known error patterns in the message
    elif 'rate limit' in exc_str or 'quota' in exc_str:
        message = ERROR_MESSAGES['RateLimitError']
    elif 'timeout' in exc_str:
        message = ERROR_MESSAGES['TimeoutError']
    elif 'connection' in exc_str or 'network' in exc_str:
        message = ERROR_MESSAGES['ConnectionError']
    elif 'authentication' in exc_str or 'api key' in exc_str or 'unauthorized' in exc_str:
        message = ERROR_MESSAGES['AuthenticationError']
    elif 'pdf' in exc_str:
        message = ERROR_MESSAGES['pdf']
    else:
        message = ERROR_MESSAGES['default']

    # Add context if provided
    if context:
        return f"{message} (while {context})"
    return message

def extract_simple_query(original_query: str) -> str:
    """
    Extract core medical terms from natural language query for fallback search.

    Creates a simpler, less restrictive query when AI-optimized query returns 0 results.
    """
    # Common stop words to remove
    stop_words = {
        'in', 'with', 'how', 'do', 'does', 'what', 'when', 'where', 'for',
        'and', 'or', 'the', 'a', 'an', 'to', 'of', 'are', 'is', 'was', 'were',
        'have', 'has', 'had', 'compare', 'versus', 'vs', 'between', 'among'
    }

    # Remove question marks and normalize
    text = original_query.lower().replace('?', '').replace(',', ' ')

    # Extract quoted phrases first (these are important)
    quoted = re.findall(r'"([^"]+)"', text)

    # Split into words
    words = text.split()

    # Keep capitalized acronyms from original (GOO, SEMS, etc.)
    acronyms = re.findall(r'\b[A-Z]{2,}\b', original_query)

    # Filter words: keep medical terms, remove stop words
    key_terms = []
    current_phrase = []

    for word in words:
        word_clean = re.sub(r'[^\w\s-]', '', word)  # Remove punctuation except hyphens

        if word_clean in stop_words or len(word_clean) < 3:
            if current_phrase:
                key_terms.append(' '.join(current_phrase))
                current_phrase = []
            continue

        current_phrase.append(word_clean)

    if current_phrase:
        key_terms.append(' '.join(current_phrase))

    # Combine quoted phrases, key terms, and acronyms
    all_terms = quoted + key_terms + [a.lower() for a in acronyms]

    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in all_terms:
        if term not in seen and term:
            seen.add(term)
            unique_terms.append(term)

    # Build simple query: use first 3-4 core terms with AND
    core_terms = unique_terms[:4]  # Limit to most important terms

    if len(core_terms) >= 2:
        # Multiple terms - use AND to combine
        simple_query = ' AND '.join([f'"{term}"' for term in core_terms])
    elif core_terms:
        # Single term
        simple_query = f'"{core_terms[0]}"'
    else:
        # Fallback: use first few words
        simple_query = ' '.join(original_query.split()[:5])

    return simple_query
