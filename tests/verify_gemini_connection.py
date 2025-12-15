import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings
from core.llm_utils import get_llm_client

def test_gemini_connection():
    try:
        print(f"Testing Gemini 3.0 Pro Connection...")
        print(f"Model: {settings.OPENAI_MODEL}")
        print(f"Base URL: {settings.GEMINI_BASE_URL}")
        
        # We need to verify we're actually picking up the key from .env (via settings)
        has_key = bool(settings.GOOGLE_API_KEY)
        key_suffix = settings.GOOGLE_API_KEY[-4:] if has_key else "None"
        print(f"GOOGLE_API_KEY is present: {has_key} (ends in ...{key_suffix})")

        client = get_llm_client(model=settings.OPENAI_MODEL)
        
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "user", "content": "Hello! Are you Gemini 3.0? Please confirm your version."}
            ],
            max_tokens=50
        )
        
        content = response.choices[0].message.content
        print("\n✅ SUCCESS! Connection Established.")
        print(f"Detailed Response: {content}")
        return True
    except Exception as e:
        print(f"\n❌ FAILURE: Connection Failed.")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_gemini_connection()
