import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

from part2.config import (
    CLARIN_API_KEY_ENV,
    CLARIN_BASE_URL,
    CLARIN_MODEL_ENV,
    ENV_FILE,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
)


def get_api_key() -> str:
    load_dotenv(ENV_FILE)
    key = os.environ.get(CLARIN_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"Brak klucza: skopiuj .env.example -> {ENV_FILE} i ustaw {CLARIN_API_KEY_ENV}")
    return key


def default_model() -> str | None:
    load_dotenv(ENV_FILE)
    return os.environ.get(CLARIN_MODEL_ENV, "").strip() or None


def list_models(api_key: str | None = None) -> list[dict]:
    client = OpenAI(base_url=CLARIN_BASE_URL, api_key=api_key or get_api_key())
    return [m.model_dump() for m in client.models.list()]


def get_chat_model(
    model: str | None = None,
    *,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> ChatOpenAI:
    name = model or default_model()
    if not name:
        raise ValueError("Podaj model (argument lub CLARIN_MODEL w .env), lista: list_models()")
    return ChatOpenAI(
        model=name,
        base_url=CLARIN_BASE_URL,
        api_key=get_api_key(),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=LLM_TIMEOUT_S,
        max_retries=2,
    )
