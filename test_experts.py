"""
Test script to verify expert responses are generated correctly.
Run with: python test_experts.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Check API key
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: No GEMINI_API_KEY or GOOGLE_API_KEY found in environment")
    sys.exit(1)

print(f"API Key found: {api_key[:10]}...")

from travel.travel_personas import (
    TRAVEL_EXPERTS,
    EXPERT_ICONS,
    call_travel_expert_stream,  # Use streaming version (what app uses)
    get_travel_prompts
)
from config import settings

# Test destination
DESTINATION = "Tokyo, Japan"
CONTEXT = """
## TRIP PARAMETERS
- **Destination**: Tokyo, Japan
- **Dates**: Jan 15 - Jan 22, 2025 (7 nights)
- **Travelers**: 2 adults
- **Budget**: $5,000 USD
"""

def test_single_expert(expert_name: str):
    """Test a single expert response using streaming (same as app)."""
    print(f"\n{'='*60}")
    print(f"Testing: {EXPERT_ICONS.get(expert_name, '🧭')} {expert_name}")
    print('='*60)

    try:
        full_response = ""
        finish_reason = "unknown"
        chunk_count = 0

        for chunk in call_travel_expert_stream(
            persona_name=expert_name,
            clinical_question=f"Plan a trip to {DESTINATION}",
            evidence_context=CONTEXT,
            model=settings.EXPERT_MODEL,
            openai_api_key=api_key,
            max_completion_tokens=2000  # More tokens for testing
        ):
            chunk_type = chunk.get("type")
            chunk_count += 1

            if chunk_type == "chunk":
                content = chunk.get("content", "")
                full_response += content
                # Print first few chunks for debugging
                if chunk_count <= 5:
                    print(f"  Chunk {chunk_count}: {repr(content[:50])}")
            elif chunk_type == "complete":
                finish_reason = chunk.get("finish_reason", "unknown")
                print(f"  Complete: {finish_reason}")
            elif chunk_type == "error":
                print(f"  ERROR in stream: {chunk.get('content')}")

        print(f"Total chunks received: {chunk_count}")

        print(f"Status: {finish_reason}")
        print(f"Response length: {len(full_response)} chars")
        print(f"\nFirst 500 chars:\n{full_response[:500]}...")

        # Check for emojis in response
        has_emoji = any(ord(c) > 127 for c in full_response)
        print(f"\nContains non-ASCII (likely emojis): {has_emoji}")

        return True, full_response

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_all_experts():
    """Test all experts and report results."""
    results = {}

    # Test subset for speed
    test_experts = ["Budget Advisor", "Activity Curator"]

    for expert in test_experts:
        success, content = test_single_expert(expert)
        results[expert] = {
            "success": success,
            "length": len(content) if success else 0,
            "has_content": len(content) > 100 if success else False
        }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for expert, data in results.items():
        icon = EXPERT_ICONS.get(expert, "🧭")
        status = "✅" if data["success"] and data["has_content"] else "❌"
        print(f"{status} {icon} {expert}: {data['length']} chars")

    return results


if __name__ == "__main__":
    print("Travel Expert Test")
    print(f"Model: {settings.EXPERT_MODEL}")
    print(f"Destination: {DESTINATION}")

    results = test_all_experts()

    # Exit code
    all_passed = all(r["success"] and r["has_content"] for r in results.values())
    sys.exit(0 if all_passed else 1)
