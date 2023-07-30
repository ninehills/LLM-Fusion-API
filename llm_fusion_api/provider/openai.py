import httpx
import logging
from typing import List, Dict

from starlette.requests import Request
from starlette.background import BackgroundTask
from starlette.responses import Response, StreamingResponse
from .base import Model, ChatHandler, EmbeddingHandler


logger = logging.getLogger(__name__)

class OpenAI(ChatHandler):
    def __init__(self, openai_api_base: str, openai_api_key: str, provider: str = "openai"):
        self.openai_api_base = openai_api_base
        self.openai_api_key = openai_api_key
        self.provider = provider

    def get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.openai_api_key:
            headers["Authorization"] = "Bearer " + self.openai_api_key
        return headers

    async def list_models(self) -> List[Model]:
        """List all models from OpenAI API"""
        headers = self.get_headers()
        async with httpx.AsyncClient() as client:
            response = await client.request(
                'GET',
                self.openai_api_base + '/models',
                headers=headers
            )
            data = response.json()

        result = []
        for model in data["data"]:
            if self.provider == "openai":
                if model["id"].startswith("gpt"):
                    result.append(Model(
                        provider=self.provider,
                        name=model["id"],
                        type="chat"
                    ))
                elif model["id"].startswith("text-embedding"):
                    result.append(Model(
                        provider=self.provider,
                        name=model["id"],
                        type="embedding"
                    ))
            else:
                if "embedding" in model["id"]:
                    result.append(Model(
                        provider=self.provider,
                        name=model["id"],
                        type="embedding"
                    ))
                else:
                    result.append(Model(
                        provider=self.provider,
                        name=model["id"],
                        type="chat"
                    ))
        return result

    async def proxy(self, path, body, request: Request) -> Response:
        """Proxy to OpenAI API"""
        headers = self.get_headers()
        url = self.openai_api_base + path
        logger.info(f"OpenAI Proxying request to {url}, headers: {headers}, body: {body}")

        client = httpx.AsyncClient()
        req = client.build_request(
            request.method,
            url,
            headers=headers,
            json=body, # type: ignore
        )
        res = await client.send(req, stream=True)
        res.headers['Access-Control-Allow-Origin'] = '*'

        return StreamingResponse(
            res.aiter_text(),
            status_code=res.status_code,
            headers=res.headers,
            background=BackgroundTask(res.aclose)
        )

    async def chat_completions(self, request: Request, model: str) -> Response:
        """POST /v1/chat/completions

        https://platform.openai.com/docs/api-reference/chat
        """

        body = await request.json()
        body["model"] = model
        return await self.proxy("/chat/completions", body, request)

    async def embeddings(self, request: Request, model: str) -> Response:
        """https://platform.openai.com/docs/api-reference/embeddings
        """
        body = await request.json()
        body["model"] = model
        return await self.proxy("/embeddings", body, request)
