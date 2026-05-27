import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENAI_BASE = "https://api.openai.com/v1"


def _ensure_env_loaded() -> None:
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)


class Settings(BaseModel):
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    pinecone_api_key: SecretStr | None = None

    llm_api_base: str = _OPENROUTER_BASE
    llm_api_key: SecretStr | None = None

    judge_api_base: str = _OPENROUTER_BASE
    judge_api_key: SecretStr | None = None

    embed_api_base: str = _OPENAI_BASE
    embed_api_key: SecretStr | None = None
    embed_model: str = "text-embedding-3-small"
    embed_dimensions: int = 512


def get_settings() -> Settings:
    _ensure_env_loaded()

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    raw: dict = {}

    if openai_key:
        raw["openai_api_key"] = openai_key
    if openrouter_key:
        raw["openrouter_api_key"] = openrouter_key
    if pinecone_key:
        raw["pinecone_api_key"] = pinecone_key

    llm_api_base = os.getenv("LLM_API_BASE")
    if llm_api_base:
        raw["llm_api_base"] = llm_api_base

    llm_api_key = os.getenv("LLM_API_KEY") or openrouter_key or openai_key
    if llm_api_key:
        raw["llm_api_key"] = llm_api_key

    judge_api_key = os.getenv("JUDGE_API_KEY") or openrouter_key
    if judge_api_key:
        raw["judge_api_key"] = judge_api_key

    embed_api_base = os.getenv("EMBED_API_BASE")
    if embed_api_base:
        raw["embed_api_base"] = embed_api_base

    embed_api_key = os.getenv("EMBED_API_KEY") or openai_key
    if embed_api_key:
        raw["embed_api_key"] = embed_api_key

    embed_model = os.getenv("EMBED_MODEL")
    if embed_model:
        raw["embed_model"] = embed_model

    embed_dimensions = os.getenv("EMBED_DIMENSIONS")
    if embed_dimensions:
        raw["embed_dimensions"] = int(embed_dimensions)

    return Settings.model_validate(raw)


def get_llm_model() -> str:
    _ensure_env_loaded()
    return os.getenv("LLM_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
