"""
Query Expansion for Literature Review

Uses LLM to expand queries with:
- Drug name aliases (Keytruda → Pembrolizumab, MK-3475)
- Medical synonyms (lung cancer → NSCLC)
- Abbreviation expansions
"""

import json
import logging
from typing import List

logger = logging.getLogger(__name__)


def expand_query(query: str, model: str = "gpt-5-mini") -> List[str]:
    """
    Expand query with synonyms and variants using LLM.

    Generates 2-3 query variants to improve recall, especially for:
    - Drug name aliases
    - Medical synonyms
    - Abbreviation expansions

    Args:
        query: Original user query
        model: LLM model to use for expansion

    Returns:
        List of queries (original + variants)
    """
    try:
        from core.llm_utils import get_llm_client
        client = get_llm_client(model=model)

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": """You are a pharmaceutical search query expander.
Given a query about clinical trials or drugs, generate 2-3 search query variants that include:
- Drug name aliases (brand names, generic names, research codes)
- Medical term synonyms
- Abbreviation expansions

Return ONLY a JSON array of strings, no explanation.
Example: ["original query", "variant 1", "variant 2"]"""
            }, {
                "role": "user",
                "content": f"Expand this query: {query}"
            }],
            max_completion_tokens=200,
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON array
        if result_text.startswith('['):
            variants = json.loads(result_text)
            # Ensure original query is included
            if query not in variants:
                variants.insert(0, query)
            logger.info(f"Query expanded to {len(variants)} variants")
            return variants[:4]  # Limit to 4 variants

    except ImportError:
        logger.debug("OpenAI not available for query expansion")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse expansion response: {e}")
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")

    # Return original query if expansion fails
    return [query]


def get_drug_aliases(drug_name: str, model: str = "gpt-5-mini") -> List[str]:
    """
    Get common aliases for a drug name.

    Args:
        drug_name: Drug name to look up
        model: LLM model to use

    Returns:
        List of drug name aliases
    """
    try:
        from core.llm_utils import get_llm_client
        client = get_llm_client(model=model)

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": """You are a pharmaceutical expert.
Given a drug name, return all known aliases including:
- Brand names (e.g., Keytruda, Opdivo)
- Generic names (e.g., pembrolizumab, nivolumab)
- Research codes (e.g., MK-3475, BMS-936558)

Return ONLY a JSON array of strings."""
            }, {
                "role": "user",
                "content": f"What are the aliases for: {drug_name}"
            }],
            max_completion_tokens=200,
        )

        result_text = response.choices[0].message.content.strip()

        if result_text.startswith('['):
            aliases = json.loads(result_text)
            if drug_name not in aliases:
                aliases.insert(0, drug_name)
            return aliases

    except Exception as e:
        logger.warning(f"Drug alias lookup failed: {e}")

    return [drug_name]
