import httpx
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse
from .base import ChatHandler


logger = logging.getLogger(__name__)

class OpenAI(ChatHandler):
    def __init__(self, openai_api_base: str, openai_api_key: str):
        self.openai_api_base = openai_api_base
        self.openai_api_key = openai_api_key

    async def proxy(self, request:Request, path: str) -> JSONResponse:
        """Proxy to OpenAI API"""
        headers = {
            "Authorization": "Bearer " + self.openai_api_key,
            "Content-Type": "application/json",
        }
        body = await request.body()
        logger.info(f"OpenAI Proxying request to {self.openai_api_base + path}, headers: {headers}, body: {body}")

        async with httpx.AsyncClient() as client:
            response = await client.request(
                request.method,
                self.openai_api_base + path,
                headers=headers,
                data=body # type: ignore
            )

        response.headers['Access-Control-Allow-Origin'] = '*'
        return JSONResponse(response.json(), status_code=response.status_code, headers=response.headers)

    async def chat_completions(self, request: Request) -> JSONResponse:
        """POST /v1/chat/completions

        https://platform.openai.com/docs/api-reference/chat
        """
        return await self.proxy(request, path="/chat/completions")
