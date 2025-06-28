import os
from pydantic import BaseModel
from typing import List, Optional

# API Configuration
API_PREFIX = "/api"
API_V1_STR = "/v1"
API_TITLE = "RAG Framework with DeepSeek-R"
API_DESCRIPTION = """
API for a Retrieval-Augmented Generation framework with:
- FastAPI backend
- DeepSeek-R LLM integration
- Voice input/output capabilities
- Swagger/OpenAPI documentation
"""
API_VERSION = "1.0.0"
OPENAPI_URL = f"{API_PREFIX}{API_V1_STR}/openapi.json"
DOCS_URL = f"{API_PREFIX}{API_V1_STR}/docs"
REDOC_URL = f"{API_PREFIX}{API_V1_STR}/redoc"

# DeepSeek-R Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Voice Configuration
VOICE_INPUT_ENABLED = os.getenv("VOICE_INPUT_ENABLED", "true").lower() == "true"
VOICE_OUTPUT_ENABLED = os.getenv("VOICE_OUTPUT_ENABLED", "true").lower() == "true"

# File Storage Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
ALLOWED_EXTENSIONS = ["pdf", "docx", "txt"]

# Vector Store Configuration
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")\

USE_GPU = "true"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

class Settings(BaseModel):
    # API Config
    api_prefix: str = API_PREFIX
    api_v1_str: str = API_V1_STR
    api_title: str = API_TITLE
    api_description: str = API_DESCRIPTION
    api_version: str = API_VERSION
    openapi_url: str = OPENAPI_URL
    docs_url: str = DOCS_URL
    redoc_url: str = REDOC_URL
    
    # DeepSeek Config
    deepseek_api_key: str = DEEPSEEK_API_KEY
    deepseek_base_url: str = DEEPSEEK_BASE_URL
    deepseek_model: str = DEEPSEEK_MODEL
    
    # Voice Config
    voice_input_enabled: bool = VOICE_INPUT_ENABLED
    voice_output_enabled: bool = VOICE_OUTPUT_ENABLED
    
    # Storage Config
    data_dir: str = DATA_DIR
    allowed_extensions: List[str] = ALLOWED_EXTENSIONS
    vector_store_dir: str = VECTOR_STORE_DIR

    # GPU Config
    use_gpu: bool = USE_GPU.lower() == "true"

settings = Settings()
