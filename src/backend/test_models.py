# backend/test_models.py
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

def test_cohere_models():
    """Test which Cohere models are currently available"""
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    models_to_test = [
        "command",
        "command-light",
        "command-r7b-12-2024",
        "command-r7b-12-2024-4bit",
        "c4ai-command-r-v01",  # Another possible model
        "c4ai-command-r-plus",  # Another possible model
    ]
    
    print("🧪 Testing Cohere models...")
    
    for model in models_to_test:
        try:
            print(f"\nTesting model: {model}")
            response = co.chat(
                message="Hello, are you working?",
                model=model,
                temperature=0.1,
                max_tokens=10
            )
            print(f"✅ SUCCESS: {model}")
            print(f"   Response: {response.text}")
        except Exception as e:
            print(f"❌ FAILED: {model}")
            print(f"   Error: {str(e)[:100]}")

if __name__ == "__main__":
    test_cohere_models()