import json
import time
import httpx
import logging
from typing import List

from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from sse_starlette.sse import EventSourceResponse

from llm_fusion_api.provider.base import ChatHandler, Model, EmbeddingHandler
from llm_fusion_api.response import ErrorResponse


logger = logging.getLogger(__name__)

MODEL_ENDPOINT_MAP = {
    "ernie-bot": "completions",
    "ernie-bot-turbo": "eb-instant",
    "ernie-bot-4": "completions_pro",
    "ernie-bot-8k": "ernie_bot_8k",
    "ernie-speed": "ernie_speed",
}

class Wenxin(ChatHandler, EmbeddingHandler):
    cached_token: str = ""
    cached_token_expires_at: int = 0

    def __init__(self, wenxin_api_key: str, wenxin_secret_key: str):
        self.wenxin_api_key = wenxin_api_key
        self.wenxin_secret_key = wenxin_secret_key

    async def list_models(self) -> List[Model]:
        """List all models from Wenxin API"""
        chat_models = [Model(provider="wenxin", name=name, type="chat") for name in MODEL_ENDPOINT_MAP]
        embedding_models = [
            Model(provider="wenxin", name="embedding-v1", type="embedding"),
            Model(provider="wenxin", name="bge_large_zh", type="embedding"),
            Model(provider="wenxin", name="bge_large_en", type="embedding"),
            Model(provider="wenxin", name="tao_8k", type="embedding"),
            ]
        return chat_models + embedding_models

    async def get_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials" +\
            f"&client_id={self.wenxin_api_key}&client_secret={self.wenxin_secret_key}"

        if self.cached_token and self.cached_token_expires_at > time.time():
            return self.cached_token

        async with httpx.AsyncClient() as client:
            response = await client.get(url=url)
            data = response.json()
            logger.info(f"Wenxin token request: {data}")
            if "error" in data:
                logger.error(f"Wenxin token error: {data['error']}")
                raise Exception(f"Wenxin token error: {data['error']}")
            self.cached_token = str(data["access_token"])
            # Wenxin token expires in 30 days, but we will refresh it every 29 days
            self.cached_token_expires_at = time.time() + data["expires_in"] - 24 * 3600

        return self.cached_token

    async def chat_completions(self, request: Request, model: str) -> Response:
        """https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11
        """
        token = await self.get_token()
        body = await request.json()
        new_body = convert_request(body)

        endpoint = MODEL_ENDPOINT_MAP.get(model.lower(), model)
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{endpoint}"
        stream = body.get('stream', False)
        kwargs = dict(
            url=url,
            headers={"Content-Type": "application/json"},
            params={"access_token": token},
            json=new_body
        )
        logger.info(f"Wenxin request to {kwargs}")

        if not stream:
            async with httpx.AsyncClient() as client:
                response: httpx.Response = await client.post(**kwargs, timeout=600) # type: ignore

            response.raise_for_status()
            res_body = response.json()
            error_code = res_body.get("error_code", None)
            if error_code:
                if error_code == 110 or error_code == 111:
                    # Token expired
                    self.cached_token = ""
                logger.error(f"Wenxin error: {error_code}")
                return ErrorResponse(500, f"Wenxin error: {error_code}")
            logger.info(f"Wenxin response: {res_body}")
            return JSONResponse(convert_response(res_body, model))

        # stream mode
        async def stream_generator():
            client = httpx.AsyncClient()
            first = True
            completion_tokens = [0]
            async with client.stream(method='POST', **kwargs) as response: # type: ignore
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:].strip())
                    logger.info(f"Wenxin stream response: {payload}")
                    if first:
                        first = False
                        first_payload = {
                            "id": payload["id"],
                            "created": payload["created"],
                        }
                        yield convert_sse_response(first_payload, model, completion_tokens)
                    yield convert_sse_response(payload, model, completion_tokens)
            yield "[DONE]"

        r = EventSourceResponse(stream_generator())
        r.ping_interval = 9999999
        return r

    async def embeddings(self, request: Request, model: str) -> Response:
        """https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu
        """
        body = await request.json()
        if isinstance(body['input'], str):
            inputs = [body['input']]
        else:
            inputs = body['input']
        for i, input in enumerate(inputs):
            # Wenxin only supports 384 tokens
            inputs[i] = input[:384]
        new_body = {'input': inputs}

        kwargs = dict(
            url=f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/{model}",
            headers={"Content-Type": "application/json"},
            params={"access_token": await self.get_token()},
            json=new_body
        )
        logger.info(f"Wenxin request to {kwargs}")
        async with httpx.AsyncClient() as client:
            response: httpx.Response = await client.post(**kwargs) # type: ignore

        response.raise_for_status()
        res_body = response.json()
        error_code = res_body.get("error_code", None)
        if error_code:
            if error_code == 110 or error_code == 111:
                # Token expired
                self.cached_token = ""
            logger.error(f"Wenxin error: {error_code}")
            return ErrorResponse(500, f"Wenxin error: {error_code}")
        logger.info(f"Wenxin response: {res_body}")
        return JSONResponse({
            'model': model,
            'object': 'list',
            'usage': res_body['usage'],
            'data': res_body['data']
        })


def convert_request(body):
    """Convert OpenAI request body to Wenxin format"""
    msg = []
    system = ""
    if body['messages'][0]['role'] == 'system':
        system = body['messages'][0]['content']
        msg = body['messages'][1:]
    else:
        msg = body['messages']

    new =  {
        'stream': body.get('stream', False),
        'messages': msg,
    }
    if "temperature" in body:
        # Wenxin temperature is between 0.001 and 1
        new["temperature"] = min(max(body["temperature"], 0.001), 1)
    if system != "":
        new["system"] = system
    if "max_tokens" in body:
        new["max_output_tokens"] = body["max_tokens"]

    return new

def convert_response(body, model):
    """Convert Wenxin response to OpenAI format"""
    return {
        'id': body['id'],
        'object': "chat.completion",
        'created': body['created'],
        'model': model,
        'choices': [
            {
                'finish_reason': "stop",
                'index': 0,
                'message': {
                    'role': "assistant",
                    'content': body['result'],
                },
            }
        ],
        'finish_reason': "stop",
        'index': 0,
        'usage': body['usage'],
    }

def convert_sse_response(body, model, last_completion_tokens):
    """Convert Wenxin SSE response to OpenAI format"""
    response =  {
        'id': body['id'],
        'object': "chat.completion.chunk",
        'created': body['created'],
        'model': model,
        'choices': [
            {
                'index': 0,
                'delta': {},
                'finish_reason': None,
            }
        ]
    }
    if body.get('usage'):
        response['usage'] = body['usage']

    if body.get('is_end'):
        # stream end
        response['choices'][0]['finish_reason'] = "stop"
        response['choices'][0]['delta'] = {
            'content': body['result'],
        }
    elif not body.get('result'):
        response['choices'][0]['delta'] = {
            'role': "assistant",
            "content": "",
        }
    else:
        response['choices'][0]['delta'] = {
            'content': body['result'],
        }
    return json.dumps(response, ensure_ascii=False).strip()
