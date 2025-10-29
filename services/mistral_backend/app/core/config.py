import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized configuration management"""

    # Model Configuration
    MODEL_BASE_PATH: str = "/home/fishy/Data/College/College-Projects/pdf_analyzer_airgapped/shared/models/"
    MODEL_PATH: str = MODEL_BASE_PATH + "mistral-7b-v0.1.Q4_0.gguf"
    N_CTX: int = int(os.getenv("N_CTX", "4096"))
    N_GPU_LAYERS: int = int(os.getenv("N_GPU_LAYERS", "0"))
    N_THREADS: int = int(os.getenv("N_THREADS", "8"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    TOP_P: float = float(os.getenv("TOP_P", "0.95"))

    # Cache Configuration
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "100"))
    CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "24"))

    # Chat History Configuration (New)
    CHAT_HISTORY_MAX_SIZE: int = int(os.getenv("CHAT_HISTORY_MAX_SIZE", "50"))
    CHAT_HISTORY_TTL_HOURS: int = int(os.getenv("CHAT_HISTORY_TTL_HOURS", "2"))

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv(
        "RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

    # API Configuration
    API_TITLE: str = "Insurance Policy Summarization API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    class Config:
        # This allows pydantic-settings to load from a .env file
        env_file = ".env"
        env_file_encoding = 'utf-8'


# Create a single, importable instance of the settings
settings = Settings()
