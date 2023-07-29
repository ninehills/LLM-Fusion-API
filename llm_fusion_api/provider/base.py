from abc import ABC, abstractmethod

from starlette.requests import Request
from starlette.responses import Response

class Model(object):
    provider: str
    name: str
    type: str

    def __init__(self, provider: str, name: str, type: str):
        self.provider = provider
        self.name = name
        self.type = type


class ChatHandler(ABC):
    @abstractmethod
    async def chat_completions(self, request: Request) -> Response:
        """POST /v1/chat/completions

        https://platform.openai.com/docs/api-reference/chat
        """
        pass


class EmbeddingHandler(ABC):
    @abstractmethod
    async def embeddings(self, request: Request) -> Response:
        """POST /v1/embeddings

        https://platform.openai.com/docs/api-reference/embeddings
        """
        pass
