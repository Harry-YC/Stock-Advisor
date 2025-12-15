"""
Adaptive Query Parser for PubMed Literature Search

Converts natural language research questions into optimized PubMed search queries.
Uses GPT-5-mini for intelligent parsing with fallback strategies.

Features:
- PICO-based parsing for comparative effectiveness questions
- Concept-based fallback for exploratory searches
- Medical term extraction and synonym expansion
- Graceful degradation when AI unavailable

Usage:
    from core.query_parser import AdaptiveQueryParser

    parser = AdaptiveQueryParser(openai_api_key="sk-...")
    result = parser.parse("In adults with GOO, how do SEMS compare to surgery?")

    print(result['optimized_query'])
    print(result['explanation'])
"""

import json
import logging
import re
from typing import Dict, List, Optional

# Set up logger for this module
logger = logging.getLogger("literature_review.query_parser")
from dataclasses import dataclass

from config import settings
from core.llm_utils import get_llm_client


@dataclass
class QueryResult:
    """Result from query parsing"""
    optimized_query: str
    query_type: str  # "PICO", "CONCEPT", "SIMPLE"
    explanation: str
    components: Dict
    confidence: str  # "high", "medium", "low"


class AdaptiveQueryParser:
    """
    Intelligent query parser that adapts to different query types.

    Supports domain-specific context injection for palliative surgery.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = None,
        domain: Optional[str] = "palliative_surgery"
    ):
        """
        Initialize parser

        Args:
            openai_api_key: API key (optional, falls back to simpler parsing if missing)
            model: Model to use (default: settings.EXPERT_MODEL for Gemini compatibility)
            domain: Domain for context injection (default: "palliative_surgery", None to disable)
        """
        self.api_key = openai_api_key
        # Use EXPERT_MODEL (Gemini) as default since it works with Google API key
        self.model = model or settings.EXPERT_MODEL
        self.ai_available = bool(openai_api_key or settings.OPENAI_API_KEY or settings.GOOGLE_API_KEY)

        # Domain-specific vocabulary
        self.domain = domain
        self.domain_filter = None
        self.domain_context = ""
        self.domain_config = None

        if domain:
            # Try to load from new DomainConfig system first (v2)
            try:
                from config.domain_config import get_domain_config
                self.domain_config = get_domain_config(domain)
                if self.domain_config:
                    # Use DomainConfig's union filter and LLM context
                    self.domain_filter = self.domain_config.get_union_filter()
                    self.exclusion_filter = self.domain_config.get_exclusion_filter()
                    self.domain_context = self.domain_config.llm_context
                    logger.info(f"Loaded domain config: {domain}")
            except ImportError:
                logger.warning("domain_config module not available, trying legacy vocabulary")

            # Fallback to legacy vocabulary if DomainConfig not available
            if not self.domain_config and domain == "palliative_surgery":
                try:
                    from config.palliative_surgery_vocabulary import (
                        DEFAULT_DOMAIN_FILTER,
                        EXCLUSION_FILTER,
                        DOMAIN_KEYWORDS,
                    )
                    self.domain_filter = DEFAULT_DOMAIN_FILTER.strip()
                    self.exclusion_filter = EXCLUSION_FILTER.strip()
                    self.domain_keywords = DOMAIN_KEYWORDS
                    self.domain_context = self._build_domain_context()
                except ImportError:
                    logger.warning("Palliative surgery vocabulary not available")
                    self.domain = None

        if self.ai_available:
            try:
                self.client = get_llm_client(api_key=openai_api_key, model=self.model)
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
                self.ai_available = False

    def _build_domain_context(self) -> str:
        """Build domain context string for prompt injection."""
        return """
IMPORTANT: This search is for PALLIATIVE SURGERY guideline development.
Focus on:
- Palliative (not curative) surgical and interventional procedures
- Symptom management in advanced/metastatic cancer
- Quality of life outcomes for terminal patients
- Comparison of surgical vs non-surgical palliation options

Highly relevant topics:
- Malignant bowel obstruction, gastric outlet obstruction
- Pathologic fractures, bone metastases
- Malignant pleural effusion, ascites
- Airway obstruction, bleeding control
- Biliary obstruction from malignancy

Key MeSH terms to consider:
- Palliative Care[MeSH], Terminal Care[MeSH]
- Quality of Life[MeSH], Symptom Assessment[MeSH]
- Plus condition-specific MeSH (e.g., Gastric Outlet Obstruction[MeSH])

NOT relevant (exclude or de-prioritize):
- Curative surgery, adjuvant/neoadjuvant treatment
- Pediatric populations (unless specifically asked)
- Benign conditions, congenital diseases
- Animal models, in vitro studies
"""

    def parse(self, query: str) -> QueryResult:
        """
        Parse natural language query into optimized PubMed syntax

        Args:
            query: Natural language research question

        Returns:
            QueryResult with optimized query and metadata
        """
        # Clean input
        query = query.strip()

        if not query:
            return QueryResult(
                optimized_query="",
                query_type="EMPTY",
                explanation="No query provided",
                components={},
                confidence="low"
            )

        # Check if query already looks like PubMed syntax
        if self._is_pubmed_syntax(query):
            return QueryResult(
                optimized_query=query,
                query_type="DIRECT",
                explanation="Query already uses proper PubMed syntax",
                components={"original": query},
                confidence="high"
            )

        # Use AI if available
        if self.ai_available:
            try:
                return self._ai_parse(query)
            except Exception as e:
                logger.warning(f"AI parsing failed: {e}, falling back to rule-based")

        # Fallback to simple rule-based parsing
        return self._simple_parse(query)

    def _is_pubmed_syntax(self, query: str) -> bool:
        """Check if query already uses PubMed syntax"""
        indicators = [
            r'\[MeSH\]',
            r'\[tiab\]',
            r'\[Title\]',
            r'\[Author\]',
            r'\[Date - Publication\]',
        ]
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in indicators)

    def _ai_parse(self, query: str) -> QueryResult:
        """
        Use GPT-5-mini to parse query intelligently
        """
        # Step 1: Detect query type
        query_type = self._detect_query_type(query)

        # Step 2: Parse based on type
        if query_type == "PICO":
            return self._pico_pipeline(query)
        elif query_type == "CONCEPT":
            return self._concept_pipeline(query)
        else:
            return self._simple_pipeline(query)

    def _detect_query_type(self, query: str) -> str:
        """
        Use GPT-5-mini to classify query type
        """
        prompt = f"""Analyze this medical literature search query and classify it:

Query: "{query}"

Classify as ONE of:
- PICO: Has clear Population, Intervention, Comparison, and Outcome (comparative effectiveness question)
- CONCEPT: Has medical concepts but no comparison (exploratory or descriptive)
- SIMPLE: Just keywords or a simple phrase

Respond with ONLY the classification word (PICO, CONCEPT, or SIMPLE)."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )

            classification = response.choices[0].message.content.strip().upper()

            if classification in ["PICO", "CONCEPT", "SIMPLE"]:
                return classification
            return "CONCEPT"  # Default fallback

        except Exception as e:
            print(f"Query type detection failed: {e}")
            return "CONCEPT"

    def _pico_pipeline(self, query: str) -> QueryResult:
        """
        Full PICO extraction and query building
        """
        # Inject domain context if available
        domain_section = ""
        if self.domain_context:
            domain_section = f"\n{self.domain_context}\n"

        prompt = f"""You are a PubMed search expert. Extract PICO components from this research question and generate an optimized PubMed search query.
{domain_section}
Research Question: "{query}"

Extract:
1. Population (P): Target patient group
2. Intervention (I): Main intervention/exposure
3. Comparison (C): Comparator (if any)
4. Outcome (O): Outcomes of interest

For each component, identify:
- Main medical terms
- Synonyms and related terms
- MeSH headings (if known)
- Acronyms

Then build a PubMed query using:
- MeSH terms with [MeSH] tag
- Title/abstract terms with [tiab] tag
- Boolean operators (AND, OR, NOT)
- Proper parentheses for grouping

Respond in JSON format:
{{
    "population": {{"terms": [...], "mesh": [...], "synonyms": [...]}},
    "intervention": {{"terms": [...], "mesh": [...], "synonyms": [...]}},
    "comparison": {{"terms": [...], "mesh": [...], "synonyms": [...]}},
    "outcome": {{"terms": [...], "mesh": [...], "synonyms": [...]}},
    "pubmed_query": "...",
    "explanation": "..."
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return QueryResult(
                    optimized_query=data.get('pubmed_query', ''),
                    query_type="PICO",
                    explanation=data.get('explanation', 'PICO-based query generated'),
                    components=data,
                    confidence="high"
                )

        except Exception as e:
            logger.warning(f"PICO pipeline failed: {e}")

        # Fallback to concept pipeline
        return self._concept_pipeline(query)

    def _concept_pipeline(self, query: str) -> QueryResult:
        """
        Simpler concept-based parsing for non-PICO queries
        """
        # Inject domain context if available
        domain_section = ""
        if self.domain_context:
            domain_section = f"\n{self.domain_context}\n"

        prompt = f"""You are a PubMed search expert. Convert this query into an optimized PubMed search.
{domain_section}
Query: "{query}"

Extract key medical concepts and generate a PubMed query using:
- MeSH terms with [MeSH] where appropriate
- Title/abstract terms with [tiab] for specific phrases
- Boolean OR for synonyms
- Boolean AND to combine different concepts

Respond in JSON format:
{{
    "concepts": [
        {{"term": "...", "mesh": "...", "synonyms": [...]}}
    ],
    "pubmed_query": "...",
    "explanation": "..."
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return QueryResult(
                    optimized_query=data.get('pubmed_query', ''),
                    query_type="CONCEPT",
                    explanation=data.get('explanation', 'Concept-based query generated'),
                    components=data,
                    confidence="medium"
                )

        except Exception as e:
            logger.warning(f"Concept pipeline failed: {e}")

        # Fallback to simple pipeline
        return self._simple_pipeline(query)

    def _simple_pipeline(self, query: str) -> QueryResult:
        """
        Direct conversion with minimal processing
        """
        prompt = f"""Convert this search query to PubMed syntax. Keep it simple.

Query: "{query}"

Return a PubMed search query using basic boolean operators and [tiab] for phrases. Keep it concise.

Respond with just the PubMed query, no explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )

            optimized = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

            return QueryResult(
                optimized_query=optimized,
                query_type="SIMPLE",
                explanation="Simple query conversion",
                components={"original": query},
                confidence="low"
            )

        except Exception as e:
            logger.warning(f"Simple pipeline failed: {e}")
            return self._simple_parse(query)

    def _simple_parse(self, query: str) -> QueryResult:
        """
        Rule-based fallback when AI is unavailable
        """
        # Basic processing: wrap phrases in quotes, add [tiab] to phrases
        words = query.split()

        # If it's a short query (< 5 words), just wrap in quotes and add [tiab]
        if len(words) <= 5:
            optimized = f'"{query}"[tiab]'
            explanation = "Simple phrase search in title/abstract"
        else:
            # Extract potential medical terms (capitalized words, acronyms)
            terms = []
            current_phrase = []

            for word in words:
                # Skip common words
                if word.lower() in ['in', 'with', 'how', 'do', 'does', 'what', 'when', 'where', 'for', 'and', 'or', 'the', 'a', 'an']:
                    if current_phrase:
                        terms.append(' '.join(current_phrase))
                        current_phrase = []
                    continue

                current_phrase.append(word)

            if current_phrase:
                terms.append(' '.join(current_phrase))

            # Build query with OR between terms
            if terms:
                optimized = ' OR '.join([f'"{term}"[tiab]' for term in terms[:5]])  # Limit to 5 terms
                explanation = "Extracted key terms combined with OR"
            else:
                optimized = f'"{query}"[tiab]'
                explanation = "Fallback to simple phrase search"

        return QueryResult(
            optimized_query=optimized,
            query_type="SIMPLE",
            explanation=explanation,
            components={"original": query},
            confidence="low"
        )

    def add_clinical_filter(
        self,
        query: str,
        category: str,
        scope: str
    ) -> str:
        """
        Append PubMed Clinical Queries filter.

        Args:
            query: Existing Boolean query
            category: Clinical question type (Therapy, Diagnosis, Prognosis, Etiology, Clinical Prediction)
            scope: "Narrow" or "Broad"

        Returns:
            Query with clinical filter appended
        """
        if not category or category == "None":
            return query

        # Extract scope keyword
        scope_key = "narrow" if "narrow" in scope.lower() else "broad"

        # PubMed Clinical Queries syntax
        clinical_filter = f"{category}/{scope_key.capitalize()}[filter]"

        # Wrap query in parens if it contains OR operators
        if " OR " in query.upper():
            return f"({query}) AND {clinical_filter}"
        else:
            return f"{query} AND {clinical_filter}"

    def add_quality_gate(self, query: str) -> str:
        """
        Exclude retractions and problematic records.

        Args:
            query: Existing Boolean query

        Returns:
            Query with exclusion filter
        """
        exclusions = (
            "NOT (Retracted Publication[pt] OR "
            "Retraction of Publication[pt] OR "
            "Expression of Concern[pt])"
        )

        return f"({query}) {exclusions}"

    def apply_domain_filter(self, query: str, require_domain_terms: bool = True) -> str:
        """
        Apply domain-specific filter to ensure palliative surgery relevance.

        Args:
            query: Existing Boolean query
            require_domain_terms: If True, AND the domain filter (strict).
                                  If False, only add if query lacks domain terms (soft).

        Returns:
            Query with domain filter applied
        """
        if not self.domain_filter:
            return query

        # Check if query already has domain terms
        domain_indicators = [
            "palliative", "palliation", "terminal", "symptom",
            "end of life", "end-of-life", "hospice", "advanced cancer",
            "metastatic", "inoperable", "unresectable"
        ]
        query_lower = query.lower()
        has_domain_terms = any(term in query_lower for term in domain_indicators)

        if has_domain_terms and not require_domain_terms:
            # Query already domain-specific, don't add redundant filter
            return query

        # Apply domain filter
        return f"({query}) AND {self.domain_filter}"

    def get_domain_info(self) -> Dict:
        """
        Get information about the current domain configuration.

        Returns:
            Dict with domain name, filter, and keywords
        """
        return {
            "domain": self.domain,
            "domain_filter": self.domain_filter,
            "has_domain_context": bool(self.domain_context),
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    print("Adaptive Query Parser - Test")
    print("=" * 70)

    # Test cases
    test_queries = [
        "In adults with unresectable malignant Gastric Outlet Obstruction (GOO), how do duodenal SEMS, EUS-GE, and surgical GJ compare for clinical success?",
        "gastric outlet obstruction treatment",
        "breast cancer",
        "machine learning radiology"
    ]

    # Use Google API key (Gemini) or OpenAI key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    parser = AdaptiveQueryParser(openai_api_key=api_key)

    for query in test_queries:
        print(f"\nOriginal: {query[:80]}...")
        result = parser.parse(query)
        print(f"Type: {result.query_type}")
        print(f"Optimized: {result.optimized_query[:150]}...")
        print(f"Confidence: {result.confidence}")
        print(f"Explanation: {result.explanation}")
        print("-" * 70)
