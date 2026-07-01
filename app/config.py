import os
from dotenv import load_dotenv

# Load .env file from backend root if it exists
backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(backend_root, ".env"))

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    VECTOR_STORE: str = os.getenv("VECTOR_STORE", "sqlite")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./memory.db")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()
