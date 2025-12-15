"""
AI-Assisted Literature Screening for Palliative Surgery GDG

Uses OpenAI API to automatically screen papers for relevance to palliative surgery
guideline development and clinical questions.
"""

import json
import logging
import time
from typing import Dict, List, Tuple, Callable, Optional


# Set up logger for this module

# Set up logger for this module
logger = logging.getLogger("literature_review.ai_screener")


def screen_single_paper(
    paper_title: str,
    paper_abstract: str,
    search_query: str,
    openai_api_key: str,
    model: str = "gpt-5-mini",
    confidence_threshold: int = 80,
    max_retries: int = 3
) -> Tuple[str, int, str]:
    """
    Screen a single paper using AI for palliative surgery relevance

    Args:
        paper_title: Title of the paper
        paper_abstract: Abstract text
        search_query: Original search query used
        openai_api_key: OpenAI API key
        model: Model to use (default: gpt-5-mini)
        confidence_threshold: Threshold for auto-decision (default: 80)
        max_retries: Maximum retries for rate limit errors (default: 3)

    Returns:
        Tuple of (decision, confidence, reasoning)
        - decision: "include", "exclude", or "review"
        - confidence: 0-100
        - reasoning: Brief explanation
    """
    from core.llm_utils import get_llm_client
    client = get_llm_client(api_key=openai_api_key)

    prompt = f"""You are screening papers for a palliative surgery guideline development literature review.

**Search Query:** {search_query}

**Paper Title:** {paper_title}

**Abstract:** {paper_abstract if paper_abstract else "[No abstract available]"}

**Task:** Determine if this paper should be INCLUDED or EXCLUDED based on relevance to the research question.

**CONTEXT - Palliative Surgery Guideline Development:**
- We are developing evidence-based guidelines for palliative surgical interventions
- Focus on: malignant bowel obstruction, pathologic fractures, bleeding control, airway obstruction
- Study types of interest: RCTs, cohort studies, case series, systematic reviews, meta-analyses
- Outcomes of interest: symptom relief, quality of life, survival, complications, reintervention rates
- Both surgical and non-surgical alternatives (stents, embolization) are relevant for comparison

**Response Format (JSON only):**
{{
  "decision": "include" or "exclude",
  "confidence": 0-100 (integer),
  "reasoning": "One sentence explaining why (max 15 words)"
}}

**Include if the paper:**
- Reports outcomes of palliative surgical procedures (bypass, diversion, resection)
- Compares surgical vs non-surgical management (stents, conservative care)
- Provides data on patient selection criteria (performance status, frailty, prognosis)
- Reports quality of life outcomes or symptom control rates
- Describes complications, mortality, or morbidity of palliative procedures
- Addresses goals of care discussions or shared decision-making in this population

**Exclude if:**
- Curative intent surgery only (no palliative focus)
- Wrong clinical scenario (clearly not relevant to query)
- Animal studies or laboratory research without clinical relevance
- Pure methodology papers without clinical data
- Retracted or erratum notice
- Not in English (unless exceptionally relevant)

Respond with ONLY the JSON object, no other text."""

    # Retry loop for rate limiting
    for attempt in range(max_retries):
        try:
            api_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert surgical oncologist screening papers for a palliative surgery guideline development review. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_completion_tokens": 150,
                "response_format": {"type": "json_object"}  # Guarantee valid JSON
            }

            response = client.chat.completions.create(**api_params)

            content = response.choices[0].message.content.strip()

            # With JSON mode, response should always be valid JSON
            result = json.loads(content)

            decision = result.get("decision", "review").lower()
            confidence = int(result.get("confidence", 50))
            reasoning = result.get("reasoning", "No reasoning provided")

            # If confidence < threshold, flag for manual review
            if confidence < confidence_threshold:
                decision = "review"

            return (decision, confidence, reasoning)

        except openai.RateLimitError as e:
            # Retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Rate limited, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Rate limited after {max_retries} retries: {e}")
                return ("review", 0, f"Rate limit exceeded: {str(e)}")

        except openai.APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            return ("review", 0, f"Connection error: {str(e)}")

        except openai.APIStatusError as e:
            logger.error(f"OpenAI API error (status {e.status_code}): {e}")
            return ("review", 0, f"API error ({e.status_code}): {str(e)}")

        except json.JSONDecodeError as e:
            # Should be rare with JSON mode, but handle it
            logger.warning(f"JSON parse error for paper '{paper_title[:50]}...': {e}")
            return ("review", 0, f"JSON parse error: {str(e)}")

        except ValueError as e:
            # Handle invalid values in response (e.g., non-integer confidence)
            logger.warning(f"Invalid response format: {e}")
            return ("review", 0, f"Invalid response: {str(e)}")

    # Should not reach here, but just in case
    return ("review", 0, "Max retries exceeded")


def screen_papers_batch(
    papers: List[Dict],
    search_query: str,
    openai_api_key: str,
    model: str = "gpt-5-mini",
    confidence_threshold: int = 80,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict]:
    """
    Screen multiple papers using AI

    Args:
        papers: List of paper dicts with 'title', 'abstract', 'pmid'
        search_query: Original search query
        openai_api_key: OpenAI API key
        model: Model to use
        confidence_threshold: Threshold for auto-decision
        progress_callback: Optional function to call with progress (current, total)

    Returns:
        List of dicts with added keys: ai_decision, ai_confidence, ai_reasoning
    """
    results = []
    total = len(papers)

    for i, paper in enumerate(papers):
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')

        decision, confidence, reasoning = screen_single_paper(
            paper_title=title,
            paper_abstract=abstract,
            search_query=search_query,
            openai_api_key=openai_api_key,
            model=model,
            confidence_threshold=confidence_threshold
        )

        result = paper.copy()
        result['ai_decision'] = decision
        result['ai_confidence'] = confidence
        result['ai_reasoning'] = reasoning

        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

    return results


def get_screening_summary(papers: List[Dict]) -> Dict[str, int]:
    """
    Get summary statistics of AI screening results

    Args:
        papers: List of papers with ai_decision field

    Returns:
        Dict with counts for include, exclude, review
    """
    summary = {
        "include": 0,
        "exclude": 0,
        "review": 0,
        "total": len(papers)
    }

    for paper in papers:
        decision = paper.get('ai_decision', 'review')
        if decision in summary:
            summary[decision] += 1

    return summary


def screen_for_scenario(
    papers: List[Dict],
    scenario: str,
    search_query: str,
    openai_api_key: str,
    model: str = "gpt-5-mini",
    confidence_threshold: int = 80,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_retries: int = 3
) -> List[Dict]:
    """
    Screen papers with scenario-specific criteria for palliative surgery

    Args:
        papers: List of paper dicts
        scenario: One of "surgical_outcomes", "patient_selection", "qol_symptoms", "comparative"
        search_query: Original search query
        openai_api_key: OpenAI API key
        model: Model to use
        confidence_threshold: Threshold for auto-decision
        progress_callback: Optional progress callback
        max_retries: Maximum retries for rate limit errors (default: 3)

    Returns:
        List of dicts with AI screening results
    """
    scenario_prompts = {
        "surgical_outcomes": """
Focus on SURGICAL OUTCOMES evidence:
- Include: Operative mortality, morbidity, complication rates for palliative procedures
- Include: Technical success rates, symptom resolution, reintervention rates
- Exclude: Papers not reporting surgical outcomes or purely curative procedures""",

        "patient_selection": """
Focus on PATIENT SELECTION criteria:
- Include: Performance status thresholds, frailty assessments, prognostic factors
- Include: Risk stratification tools, predictive models for surgical outcomes
- Exclude: Papers not addressing who benefits from palliative surgery""",

        "qol_symptoms": """
Focus on QUALITY OF LIFE and SYMPTOM evidence:
- Include: QoL outcomes (validated scales), symptom control rates, functional status
- Include: Days at home, caregiver burden, patient-reported outcomes
- Exclude: Papers not reporting QoL or symptom-related outcomes""",

        "comparative": """
Focus on COMPARATIVE EFFECTIVENESS:
- Include: Surgery vs stent, surgery vs conservative, head-to-head comparisons
- Include: Cost-effectiveness, resource utilization, length of stay comparisons
- Exclude: Papers studying only one intervention without comparator"""
    }

    scenario_context = scenario_prompts.get(scenario, "")

    results = []
    total = len(papers)

    # Create client once, reuse for all papers
    from core.llm_utils import get_llm_client
    client = get_llm_client(api_key=openai_api_key)

    for i, paper in enumerate(papers):
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')

        prompt = f"""You are screening papers for a palliative surgery guideline development literature review.

**Search Query:** {search_query}

**Scenario-Specific Criteria:**
{scenario_context}

**Paper Title:** {title}

**Abstract:** {abstract if abstract else "[No abstract available]"}

**Response Format (JSON only):**
{{
  "decision": "include" or "exclude",
  "confidence": 0-100 (integer),
  "reasoning": "One sentence explaining why (max 15 words)"
}}

Respond with ONLY the JSON object, no other text."""

        # Retry loop for rate limiting
        decision, confidence, reasoning = "review", 0, "Processing error"

        for attempt in range(max_retries):
            try:
                api_params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": f"You are screening papers for {scenario} evidence. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_completion_tokens": 150,
                    "response_format": {"type": "json_object"}  # Guarantee valid JSON
                }

                response = client.chat.completions.create(**api_params)
                content = response.choices[0].message.content.strip()

                # With JSON mode, response should always be valid JSON
                result_json = json.loads(content)

                decision = result_json.get("decision", "review").lower()
                confidence = int(result_json.get("confidence", 50))
                reasoning = result_json.get("reasoning", "No reasoning provided")

                if confidence < confidence_threshold:
                    decision = "review"

                break  # Success, exit retry loop

            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited on paper {i+1}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Rate limited after {max_retries} retries for paper {i+1}")
                    decision, confidence, reasoning = "review", 0, f"Rate limit exceeded"

            except openai.APIConnectionError as e:
                logger.error(f"Connection error for paper {i+1}: {e}")
                decision, confidence, reasoning = "review", 0, f"Connection error"
                break

            except openai.APIStatusError as e:
                logger.error(f"API error for paper {i+1}: {e}")
                decision, confidence, reasoning = "review", 0, f"API error ({e.status_code})"
                break

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error for paper {i+1}: {e}")
                decision, confidence, reasoning = "review", 0, f"JSON parse error"
                break

            except ValueError as e:
                logger.warning(f"Invalid response for paper {i+1}: {e}")
                decision, confidence, reasoning = "review", 0, f"Invalid response"
                break

        result = paper.copy()
        result['ai_decision'] = decision
        result['ai_confidence'] = confidence
        result['ai_reasoning'] = reasoning
        result['screening_scenario'] = scenario

        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

    return results
