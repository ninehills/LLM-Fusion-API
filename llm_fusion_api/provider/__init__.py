from typing import List

from .base import Model # noqa: F401
from .openai import OpenAI  # noqa: F401


def list_models(provider: str) -> List[Model]:
    """Get a list of all available models for a provider"""
    if provider == "openai":
        # https://platform.openai.com/docs/models/overview
        return [
            Model(provider="openai", name="gpt-turbo-3.5", type="chat"),
            Model(provider="openai", name="gpt-3.5-turbo-16k", type="chat"),
            Model(provider="openai", name="gpt-3.5-turbo-0613", type="chat"),
            Model(provider="openai", name="gpt-3.5-turbo-16k-0613", type="chat"),
            Model(provider="openai", name="gpt-4", type="chat"),
            Model(provider="openai", name="gpt-4-0613", type="chat"),
            Model(provider="openai", name="gpt-4-32k", type="chat"),
            Model(provider="openai", name="gpt-4-32k-0613", type="chat"),
            Model(provider="openai", name="text-embedding-ada-002", type="embedding"),
        ]
    raise NotImplementedError(f"Unknown provider: {provider}")
