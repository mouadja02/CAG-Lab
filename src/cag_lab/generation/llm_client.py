from dataclasses import dataclass

import litellm

from cag_lab.config import get_settings


@dataclass
class CompletionResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int


def complete(
    model: str,
    messages: list[dict[str, str]],
    api_base: str | None = None,
) -> CompletionResult:
    settings = get_settings()
    kwargs: dict = {}
    if api_base is not None:
        kwargs["api_base"] = api_base
    if model.startswith("openrouter/"):
        kwargs["api_key"] = settings.openrouter_api_key.get_secret_value()
    response = litellm.completion(model=model, messages=messages, **kwargs)

    content = response.choices[0].message.content or ""
    usage = response.usage or litellm.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    return CompletionResult(
        answer=content,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
