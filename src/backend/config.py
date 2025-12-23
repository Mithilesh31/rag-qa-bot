import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    
    # RAG Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K = 3
    
    # Models
    EMBEDDING_MODEL = "embed-english-v3.0"
    
    # Use the working model from your test
    GENERATION_MODEL = "command-r7b-12-2024"