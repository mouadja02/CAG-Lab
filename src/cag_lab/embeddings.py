from openai import OpenAI

from cag_lab.config import get_settings


def embed(
    text: str,
    model: str | None = None,
    dimensions: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> list[float]:
    settings = get_settings()

    effective_api_base = api_base or settings.embed_api_base
    effective_model = model or settings.embed_model
    effective_dims = dimensions or settings.embed_dimensions
    effective_key = api_key or (
        settings.embed_api_key.get_secret_value() if settings.embed_api_key else "none"
    )

    client = OpenAI(base_url=effective_api_base, api_key=effective_key)
    response = client.embeddings.create(
        model=effective_model,
        input=text,
        dimensions=effective_dims,
    )
    return response.data[0].embedding
