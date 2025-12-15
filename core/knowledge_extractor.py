"""
Knowledge Extractor for Literature Review Platform

Extracts structured knowledge (facts, triples) from expert panel discussions
using LLM-based extraction.

This enables the Knowledge Store to learn from each discussion session.
"""

import re
from typing import Dict, List, Optional, Any
from openai import OpenAI

from config import settings


def extract_facts_from_response(
    response_text: str,
    persona: str,
    clinical_question: str,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini"
) -> List[str]:
    """
    Extract key facts/insights from an expert response.

    Args:
        response_text: The expert's response text
        persona: The expert persona name
        clinical_question: The research question being discussed
        openai_api_key: OpenAI API key
        model: Model to use for extraction

    Returns:
        List of extracted facts (strings)
    """
    # Use centralized settings for API key
    if not openai_api_key:
        openai_api_key = settings.OPENAI_API_KEY

    if not openai_api_key:
        return []

    if len(response_text) < 100:
        return []

    try:
        client = OpenAI(api_key=openai_api_key, timeout=30.0)

        system_prompt = """You are a knowledge extraction assistant for drug development.

Extract 3-5 key factual statements from the expert response that would be valuable to remember for future discussions.

Focus on:
- Quantitative data (IC50, EC50, half-life, bioavailability, response rates)
- Mechanism of action insights
- Safety/toxicity findings
- PK/PD relationships
- Competitive intelligence (pipeline status, differentiation)
- Regulatory considerations
- Biomarker findings

Return ONLY the facts, one per line, no numbering or bullets.
Each fact should be a complete, standalone statement.
If no notable facts, return "NO_FACTS"."""

        user_prompt = f"""Expert: {persona}
Research Question: {clinical_question}

Expert Response:
{response_text[:3000]}

Extract key facts:"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()

        if content == "NO_FACTS" or not content:
            return []

        # Parse facts
        facts = [
            line.strip()
            for line in content.split('\n')
            if line.strip() and len(line.strip()) > 10
        ]

        return facts[:5]  # Limit to 5 facts

    except Exception as e:
        print(f"Fact extraction error: {e}")
        return []


def extract_triples_from_response(
    response_text: str,
    clinical_question: str,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini"
) -> List[Dict[str, Any]]:
    """
    Extract knowledge graph triples from an expert response.

    Args:
        response_text: The expert's response text
        clinical_question: The research question
        openai_api_key: OpenAI API key
        model: Model to use

    Returns:
        List of triple dicts with keys: subject, predicate, object, context
    """
    # Use centralized settings for API key
    if not openai_api_key:
        openai_api_key = settings.OPENAI_API_KEY

    if not openai_api_key:
        return []

    if len(response_text) < 100:
        return []

    try:
        client = OpenAI(api_key=openai_api_key, timeout=30.0)

        system_prompt = """You are a knowledge graph extraction assistant for drug development.

Extract Subject-Predicate-Object triples from the text. Focus on relationships involving:
- Compounds/drugs and their targets
- Compounds and their properties (IC50, selectivity, etc.)
- Compounds and indications
- Safety findings
- Competitive relationships

Format each triple as:
SUBJECT | PREDICATE | OBJECT | CONTEXT

Examples:
Compound X | inhibits | EGFR | with IC50 of 5 nM
Drug Y | shows_response_rate_of | 45% | in NSCLC patients
Target Z | is_associated_with | cardiotoxicity | in preclinical studies

Return up to 5 triples. If no clear relationships, return "NO_TRIPLES"."""

        user_prompt = f"""Research Question: {clinical_question}

Text:
{response_text[:2500]}

Extract triples:"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()

        if content == "NO_TRIPLES" or not content:
            return []

        # Parse triples
        triples = []
        for line in content.split('\n'):
            line = line.strip()
            if '|' not in line:
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                triple = {
                    "subject": parts[0],
                    "predicate": parts[1],
                    "object": parts[2],
                    "context": parts[3] if len(parts) > 3 else None
                }
                triples.append(triple)

        return triples[:5]  # Limit to 5 triples

    except Exception as e:
        print(f"Triple extraction error: {e}")
        return []


def extract_entities_from_question(
    clinical_question: str,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-5-mini"
) -> List[str]:
    """
    Extract key entities (compounds, targets, indications) from a research question.

    Used for knowledge relevance scoring.

    Args:
        clinical_question: The research question
        openai_api_key: OpenAI API key
        model: Model to use

    Returns:
        List of entity strings
    """
    # Use centralized settings for API key
    if not openai_api_key:
        openai_api_key = settings.OPENAI_API_KEY

    if not openai_api_key:
        # Fallback: simple regex extraction
        entities = []
        # Look for capitalized terms
        caps = re.findall(r'\b[A-Z][a-zA-Z0-9-]+\b', clinical_question)
        entities.extend(caps)
        # Look for common drug patterns
        drugs = re.findall(r'\b[a-z]+[ui]mab\b|\b[a-z]+[tn]ib\b', clinical_question, re.IGNORECASE)
        entities.extend(drugs)
        return list(set(entities))[:5]

    try:
        client = OpenAI(api_key=openai_api_key, timeout=30.0)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract key biomedical entities (drug names, targets, diseases, biomarkers) from the question. Return one entity per line, no numbering. Max 5 entities."
                },
                {"role": "user", "content": clinical_question}
            ],
            max_tokens=100,
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        entities = [
            line.strip()
            for line in content.split('\n')
            if line.strip() and len(line.strip()) > 1
        ]

        return entities[:5]

    except Exception:
        return []


def process_discussion_for_knowledge(
    discussion_responses: Dict[str, Dict[str, Any]],
    clinical_question: str,
    source_name: str,
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a complete expert panel discussion to extract knowledge.

    Args:
        discussion_responses: Dict[expert_name] -> response dict with 'content' key
        clinical_question: The research question
        source_name: Name for the source (e.g., "Expert Panel: Target X")
        openai_api_key: OpenAI API key

    Returns:
        Dict with:
        - facts: Dict[persona] -> List[str]
        - triples: List[Dict]
        - entities: List[str]
    """
    from core.knowledge_store import get_default_store

    store = get_default_store()

    # Extract entities from question
    entities = extract_entities_from_question(clinical_question, openai_api_key)

    all_facts: Dict[str, List[str]] = {}
    all_triples: List[Dict[str, Any]] = []

    for expert_name, response in discussion_responses.items():
        response_text = response.get('content', '')

        if not response_text or len(response_text) < 100:
            continue

        # Extract facts
        facts = extract_facts_from_response(
            response_text,
            expert_name,
            clinical_question,
            openai_api_key
        )

        if facts:
            all_facts[expert_name] = facts
            # Save to store
            store.add_knowledge(
                persona=expert_name,
                source=source_name,
                facts=facts,
                metadata={"question": clinical_question[:200]}
            )

        # Extract triples
        triples = extract_triples_from_response(
            response_text,
            clinical_question,
            openai_api_key
        )

        if triples:
            all_triples.extend(triples)
            # Save to store
            store.add_triples_batch(triples, source=source_name)

    return {
        "facts": all_facts,
        "triples": all_triples,
        "entities": entities,
        "facts_count": sum(len(f) for f in all_facts.values()),
        "triples_count": len(all_triples)
    }
