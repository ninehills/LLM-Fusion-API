import logging
from typing import List
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from llm_fusion_api import settings
from llm_fusion_api.response import ErrorResponse
from llm_fusion_api.provider import Model, OpenAI, Wenxin, MiniMax, Zhipu


logging.basicConfig(level=logging.INFO)


class App(Starlette):
    # All providers.
    providers = {}

    def __init__(self):
        routes = [
            Route("/", endpoint=self.homepage, methods=['GET']),
            Route("/v1/models", endpoint=self.get_models, methods=['GET']),
            Route("/v1/chat/completions", endpoint=self.chat_completions, methods=['POST']),
            Route("/v1/embeddings", endpoint=self.embeddings, methods=['POST']),
            Route("/v1/engines/{model_name:path}/embeddings", endpoint=self.embeddings, methods=['POST']),
        ]

        middleware = [
            Middleware(SecretTokenAuthMiddleware, secret_token=settings.SECRET_TOKEN),
        ]

        super().__init__(debug=settings.DEBUG, routes=routes, middleware=middleware)

        ## Register global variables
        self.load_variables()

    def load_variables(self):
        self.models = []
        if settings.OPENAI_API_KEY:
            self.providers['openai'] = OpenAI(settings.OPENAI_API_BASE, str(settings.OPENAI_API_KEY))
        if settings.WENXIN_API_KEY and settings.WENXIN_SECRET_KEY:
            self.providers['wenxin'] = Wenxin(str(settings.WENXIN_API_KEY), str(settings.WENXIN_SECRET_KEY))
        if settings.FASTCHAT_OPENAI_API_BASE:
            self.providers['fastchat'] = OpenAI(
                settings.FASTCHAT_OPENAI_API_BASE, str(settings.FASTCHAT_OPENAI_API_KEY), "fastchat")
        if settings.MINIMAX_GROUP_ID and settings.MINIMAX_API_KEY:
            self.providers['minimax'] = MiniMax(str(settings.MINIMAX_GROUP_ID), str(settings.MINIMAX_API_KEY))
        if settings.ZHIPU_API_KEY:
            self.providers['zhipu'] = Zhipu(str(settings.ZHIPU_API_KEY))

    async def list_models(self) -> List[Model]:
        """List all models from all providers."""
        models = []
        for provider in self.providers.values():
            models.extend(await provider.list_models())
        return models

    async def homepage(self, request):
        """GET /"""
        return JSONResponse({'hello': 'world'})

    async def get_models(self, request: Request) -> JSONResponse:
        """GET /v1/models

        https://platform.openai.com/docs/api-reference/models
        """
        models = await self.list_models()
        response = [
            {
                "created": 1677610602,
                "id": model.name if model.provider == "openai" else model.provider + "/" + model.name,
                "object": "model",
                "owned_by": model.provider,
                "permission": [
                    {
                        "created": 1680818747,
                        "id": "modelperm-fTUZTbzFp7uLLTeMSo9ks6oT",
                        "object": "model_permission",
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    },
                ],
                "root": model.name,
                "parent": None,
            }
            for model in models
        ]

        return JSONResponse(dict(data=response))

    async def chat_completions(self, request: Request) -> JSONResponse:
        """POST /v1/chat/completions

        https://platform.openai.com/docs/api-reference/chat
        """
        body = await request.json()
        model = body.get('model', '')
        if '/' in model:
            provider, model = model.split('/')
        else:
            # Default to OpenAI
            provider = 'openai'

        if provider not in self.providers:
            return ErrorResponse(400, f'Provider {provider} not found')
        return await self.providers[provider].chat_completions(request, model)

    async def embeddings(self, request: Request) -> JSONResponse:
        """POST
            /v1/embeddings
            /v1/engines/{model_name}/embeddings
        """
        body = await request.json()
        model = body.get('model', request.path_params.get('model_name'))
        if '/' in model:
            provider, model = model.split('/')
        else:
            # Default to OpenAI
            provider = 'openai'

        if provider not in self.providers:
            return ErrorResponse(400, f'Provider {provider} not found')
        return await self.providers[provider].embeddings(request, model)


class SecretTokenAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to check for a secret token in the Authorization header"""
    def __init__(self, app, secret_token=None):
        super().__init__(app)
        self.secret_token = secret_token

    async def dispatch(self, request, call_next):
        """Check for a secret token in the Authorization header"""
        if self.secret_token and request.headers.get('Authorization') != f'Bearer {self.secret_token}':
            return ErrorResponse(401, 'Unauthorized')
        response = await call_next(request)
        return response


# Starlette app
app = App()
