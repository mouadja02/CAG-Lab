import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, ValidationError

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseModel):
    openai_api_key: SecretStr
    openrouter_api_key: SecretStr
    pinecone_api_key: SecretStr


def get_settings() -> Settings:
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)

    raw = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
    }

    missing = [k for k, v in raw.items() if not v]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return Settings.model_validate(raw)
