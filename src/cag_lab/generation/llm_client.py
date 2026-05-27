from dataclasses import dataclass

from openai import OpenAI

from cag_lab.config import get_settings

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


@dataclass
class CompletionResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int


def complete(
    model: str,
    messages: list[dict[str, str]],
    api_base: str | None = None,
    api_key: str | None = None,
) -> CompletionResult:
    settings = get_settings()
    clean_model = model.removeprefix("openrouter/")

    if model.startswith("openrouter/"):
        effective_base = api_base or _OPENROUTER_BASE
        effective_key = api_key or (
            settings.judge_api_key.get_secret_value()
            if settings.judge_api_key
            else settings.openrouter_api_key.get_secret_value()
            if settings.openrouter_api_key
            else "none"
        )
    else:
        effective_base = api_base or settings.llm_api_base
        effective_key = api_key or (
            settings.llm_api_key.get_secret_value() if settings.llm_api_key else "none"
        )

    client = OpenAI(base_url=effective_base, api_key=effective_key)
    response = client.chat.completions.create(
        model=clean_model,
        messages=messages,  # type: ignore[arg-type]
    )

    content = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    return CompletionResult(
        answer=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
