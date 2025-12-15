
import os
import sys
from config import settings
from core.llm_utils import get_llm_client

def verify_adapter():
    print("Verifying Adapter...")
    
    # 1. Test Standard OpenAI Model
    print("\n--- Testing Standard OpenAI ---")
    client_gpt = get_llm_client(model="gpt-5-mini")
    print(f"Client Base URL: {client_gpt.base_url}")
    if "api.openai.com" in str(client_gpt.base_url):
        print("✅ Correctly routing to OpenAI")
    else:
        print(f"❌ Incorrect routing: {client_gpt.base_url}")

    # 2. Test Gemini Model
    print("\n--- Testing Gemini (via Adapter) ---")
    # Simulate REASONING_MODEL being gemini
    gemini_model = "gemini-3-pro-preview"
    
    # Mock settings if needed, but get_llm_client reads settings.
    # We expect base_url to be Google's
    client_gemini = get_llm_client(model=gemini_model)
    print(f"Client Base URL: {client_gemini.base_url}")
    
    expected_url = settings.GEMINI_BASE_URL
    if str(client_gemini.base_url) == str(expected_url):
        print("✅ Correctly routing to Google")
    else:
        print(f"❌ Incorrect routing. Expected {expected_url}, got {client_gemini.base_url}")
        
    # 3. Test API Key resolution
    print("\n--- Testing API Key Resolution ---")
    # We don't want to print real keys, just check source
    if client_gemini.api_key == settings.GOOGLE_API_KEY:
        print("✅ Adapter using GOOGLE_API_KEY for Gemini")
    elif client_gemini.api_key == settings.OPENAI_API_KEY:
         print("⚠️ Adapter using OPENAI_API_KEY (Fallback? Check env)")
    else:
         print("❓ Unknown Key Source")

if __name__ == "__main__":
    verify_adapter()
