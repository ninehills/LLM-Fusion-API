import json
from math import e
import time
import uuid
import httpx
import logging
from typing import List

import jwt
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from sse_starlette.sse import EventSourceResponse

from llm_fusion_api.provider.base import ChatHandler, Model
from llm_fusion_api.response import ErrorResponse


logger = logging.getLogger(__name__)

class Zhipu(ChatHandler):
    chat_completion_url_tpl: str = "https://open.bigmodel.cn/api/paas/v3/model-api/{model}/{invoke_type}"

    def __init__(self, zhipu_api_key: str):
        self.zhipu_api_key = zhipu_api_key

    def get_chat_completion_url(self, model: str, stream: bool) -> str:
        invoke_type = "sse-invoke" if stream else "invoke"
        return self.chat_completion_url_tpl.format(model=model, invoke_type=invoke_type)

    async def list_models(self) -> List[Model]:
        """List all models from Zhipu API"""
        return [
            Model(provider="zhipu", name="chatglm_pro", type="chat"),
            Model(provider="zhipu", name="chatglm_std", type="chat"),
            Model(provider="zhipu", name="chatglm_lite", type="chat"),
        ]

    def gen_token(self) -> str:
        id, secret = self.zhipu_api_key.split(".")
        payload = {
            "api_key": id,
            "exp": int(round(time.time() * 1000)) + 600 * 1000,
            "timestamp": int(round(time.time() * 1000)),
        }

        return jwt.encode(
            payload,
            secret,
            algorithm="HS256",
            headers={"alg": "HS256", "sign_type": "SIGN"},
        )


    async def chat_completions(self, request: Request) -> Response:
        """https://open.bigmodel.cn/doc/api#chatglm_pro
        """
        body = await request.json()
        logger.info(f"Zhipu request: {body}")
        new_body = convert_request(body)

        model = body.get('model', '').split('/')[-1]
        stream = body.get('stream', False)
        kwargs = dict(
            url=self.get_chat_completion_url(model, stream),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.gen_token()}",
            },
            json=new_body
        )
        logger.info(f"Zhipu request to {kwargs}")

        if not stream:
            async with httpx.AsyncClient() as client:
                response: httpx.Response = await client.post(**kwargs) # type: ignore

            response.raise_for_status()
            res_body = response.json()
            error_code = res_body.get("code", 0)
            if error_code != 200:
                logger.error(f"Zhipu error: {error_code}")
                return ErrorResponse(500, f"Zhipu error: {error_code} - {res_body.get('msg', '')}")
            logger.info(f"Zhipu response: {res_body}")
            return JSONResponse(convert_response(res_body, model))

        # stream mode
        async def stream_generator():
            id = uuid.uuid4().hex
            client = httpx.AsyncClient()
            first = True
            async with client.stream(method='POST', **kwargs) as response: # type: ignore
                async for line in response.aiter_lines():
                    if line.startswith("id: "):
                        id = line[4:].strip()
                        continue
                    if line.startswith('event: "finish"'):
                        yield convert_sse_response({"finished": True}, model, id=id)
                        break
                    if not line.startswith("data: "):
                        continue
                    text = line[6:].strip()
                    logger.info(f"Zhipu stream response: {text}")
                    if first:
                        first = False
                        yield convert_sse_response({}, model, id=id)
                    yield convert_sse_response({
                        "text": text,
                    }, model, id=id)
            yield "[DONE]"

        return EventSourceResponse(stream_generator())


def convert_request(body):
    """Convert OpenAI request body to Zhipu format"""
    msg = []
    if body['messages'][0]['role'] == 'system':
        if body['messages'][0]['content'].strip() == '':
            msg = body['messages'][1:]
        else:
            msg.append({
                'role': 'user',
                'content': body['messages'][0]['content']
            })
            msg.append({
                'role': 'assistant',
                'content': '收到'
            })
            msg.extend(body['messages'][1:])
    else:
        msg = body['messages']

    new =  {
        'incremental': True,
        'prompt': msg,
    }
    if "temperature" in body:
        # Wenxin temperature is between 0.001 and 1
        new["temperature"] = min(max(body["temperature"], 0.001), 1)

    if "top_p" in body:
        new["top_p"] = body["top_p"]

    return new

def convert_response(body, model):
    """Convert Zhipu response to OpenAI format"""
    r = {
        'id': body.get('request_id', uuid.uuid4().hex),
        'object': "chat.completion",
        'created': int(time.time()),
        'model': model,
        'choices': [],
        'usage': body['usage'],
    }
    for i, choice in enumerate(body['choices']):
        r['choices'].append({
            'finish_reason': 'stop',
            'index': i,
            'message': {
                'role': "assistant",
                'content': choice['content'],
            },
        })
    return r


def convert_sse_response(body, model, id=None):
    """Convert MiniMax SSE response to OpenAI format"""
    response =  {
        'id': id,
        'object': "chat.completion.chunk",
        'created': int(time.time()),
        'model': model,
        'choices': [
            {
                'index': 0,
                'delta': {},
                'finish_reason': None,
            }
        ],
    }
    if 'text' not in body:
        response['choices'][0]['delta'] = {
            'role': "assistant",
            "content": "",
        }
    elif 'finished' in body:
        response['choices'][0]['finish_reason'] = 'stop'
    else:
        response['choices'][0]['delta'] = {
            'content': body['text'],
        }

    return json.dumps(response, ensure_ascii=False)
