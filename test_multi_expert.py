"""
Test multi-expert execution to investigate why only one expert might respond.
"""

import sys
sys.path.insert(0, '/Users/nelsonliu/Travel Planner')

from config import settings
from travel.travel_personas import (
    TRAVEL_EXPERTS,
    TRAVEL_PRESETS,
    EXPERT_ICONS,
    call_travel_expert_stream
)

def test_multi_expert():
    """Test calling multiple experts like the app does."""

    # Get Quick Trip Planning preset (default)
    preset = TRAVEL_PRESETS["Quick Trip Planning"]
    selected_experts = preset["experts"]

    print(f"\n{'='*60}")
    print(f"Testing Quick Trip Planning preset")
    print(f"Experts to call: {selected_experts}")
    print(f"{'='*60}\n")

    destination = "Barcelona, Spain"
    context = "Trip to Barcelona, Spain. Budget: $5000. Dates: Jan 15-22, 2025."

    results = {}

    for expert_name in selected_experts:
        icon = EXPERT_ICONS.get(expert_name, "?")
        print(f"\n--- Calling {icon} {expert_name} ---")

        full_response = ""
        chunk_count = 0
        error_occurred = False

        try:
            for chunk in call_travel_expert_stream(
                persona_name=expert_name,
                clinical_question=f"Plan a trip to {destination}",
                evidence_context=context,
                model=settings.EXPERT_MODEL,
                openai_api_key=settings.GEMINI_API_KEY
            ):
                if chunk.get("type") == "chunk":
                    chunk_count += 1
                    content = chunk.get("content", "")
                    full_response += content
                    # Print first 50 chars of first chunk
                    if chunk_count == 1:
                        print(f"   First chunk: {content[:50]}...")
                elif chunk.get("type") == "error":
                    print(f"   ERROR: {chunk.get('content')}")
                    error_occurred = True
                elif chunk.get("type") == "complete":
                    print(f"   Complete: {chunk.get('finish_reason')}")

        except Exception as e:
            print(f"   EXCEPTION: {e}")
            error_occurred = True

        results[expert_name] = {
            "chunks": chunk_count,
            "length": len(full_response),
            "error": error_occurred,
            "preview": full_response[:200] if full_response else ""
        }

        print(f"   Result: {chunk_count} chunks, {len(full_response)} chars")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    success_count = 0
    for expert_name, result in results.items():
        icon = EXPERT_ICONS.get(expert_name, "?")
        status = "OK" if not result["error"] and result["chunks"] > 0 else "FAILED"
        if status == "OK":
            success_count += 1
        print(f"{icon} {expert_name}: {status} ({result['chunks']} chunks, {result['length']} chars)")

    print(f"\n{success_count}/{len(selected_experts)} experts responded successfully")

    return results


if __name__ == "__main__":
    print("Testing multi-expert execution...")
    test_multi_expert()
