import json
import uuid
import httpx
import logging
from typing import List

from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from sse_starlette.sse import EventSourceResponse

from llm_fusion_api.provider.base import ChatHandler, Model, EmbeddingHandler
from llm_fusion_api.response import ErrorResponse


logger = logging.getLogger(__name__)

class MiniMax(ChatHandler):
    chat_completion_url: str = "https://api.minimax.chat/v1/text/chatcompletion"

    def __init__(self, minimax_group_id: str, minimax_api_key: str):
        self.minimax_group_id = minimax_group_id
        self.minimax_api_key = minimax_api_key

    async def list_models(self) -> List[Model]:
        """List all models from MiniMax API"""
        return [
            Model(provider="minimax", name="abab5.5-chat", type="chat"),
            # MiniMax embedding models need `type` parameter what is not supported by OpenAI API.
            # Model(provider="minimax", name="embo-01", type="embedding"),
        ]

    async def chat_completions(self, request: Request, model: str) -> Response:
        """https://api.minimax.chat/document/guides/chat?id=6433f37294878d408fc82953
        """
        body = await request.json()
        new_body = convert_request(body)

        stream = body.get('stream', False)
        kwargs = dict(
            url=self.chat_completion_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.minimax_api_key}",
            },
            params={"GroupId": self.minimax_group_id},
            json=new_body
        )
        logger.info(f"MiniMax request to {kwargs}")

        if not stream:
            async with httpx.AsyncClient() as client:
                response: httpx.Response = await client.post(**kwargs) # type: ignore

            response.raise_for_status()
            res_body = response.json()
            error_code = res_body.get("base_resp", {}).get("status_code", 0)
            if error_code:
                logger.error(f"MiniMax error: {error_code}")
                return ErrorResponse(500, f"MiniMax error: {error_code}")
            logger.info(f"MiniMax response: {res_body}")
            return JSONResponse(convert_response(res_body, model))

        # stream mode
        async def stream_generator():
            id = uuid.uuid4().hex
            client = httpx.AsyncClient()
            first = True
            async with client.stream(method='POST', **kwargs) as response: # type: ignore
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:].strip())
                    logger.info(f"MiniMax stream response: {payload}")
                    if first:
                        first = False
                        first_payload = {
                            "created": payload["created"],
                        }
                        yield convert_sse_response(first_payload, model, id=id)
                    yield convert_sse_response(payload, model, id=id)
            yield "[DONE]"

        return EventSourceResponse(stream_generator())


def convert_request(body):
    """Convert OpenAI request body to MiniMax format"""
    mm_body =  {
        'use_standard_sse': True,
        'messages': [],
        'prompt': '',
        'role_meta': {
            'user_name': '用户',
            'bot_name': 'MM 智能助理',
        },
        'stream': body.get('stream', False),
        'model': body.get('model', '').split('/')[-1],
    }
    for msg in body['messages']:
        if msg['role'] == 'system' and msg['content'].strip() != '':
            # 根据 system message 修改预设 Prompt
            mm_body['prompt'] += msg['content'] + '\n\n'
        if msg['role'] == 'user':
            mm_body['messages'].append({
                'sender_type': 'USER',
                'text': msg['content']
            })
        elif msg['role'] == 'assistant':
            mm_body['messages'].append({
                'sender_type': 'BOT',
                'text': msg['content']
            })
    if mm_body['prompt'] == '':
        mm_body['prompt'] = ("MM 智能助理是一款由 MiniMax 自研的，没有调用其他产品的接口的大型语言模型。"
                             "MiniMax 是一家中国科技公司，一直致力于进行大模型相关的研究。")
    if 'max_tokens' in body:
        mm_body['tokens_to_generate'] = body['max_tokens']
    if body['messages'][0]['role'] == 'system' and body['messages'][0]['content'].strip() != '':
        # 根据 system message 修改预设 Prompt
        mm_body['prompt'] = body['messages'][0]['content']
    if "temperature" in body:
        # MiniMax temperature is between 0.001 and 1
        mm_body["temperature"] = min(max(body["temperature"], 0.001), 1)
    if "n" in body:
        mm_body["beam_width"] = max(min(body["n"], 4), 1)
    if "top_p" in body:
        mm_body["top_p"] = body["top_p"]
    return mm_body

def convert_response(body, model):
    """Convert MiniMax response to OpenAI format"""
    r = {
        'id': body.get('id', uuid.uuid4().hex),
        'object': "chat.completion",
        'created': body['created'],
        'model': model,
        'choices': [],
        'usage': body['usage'],
    }
    for choice in body['choices']:
        finish_reason = 'stop'
        if choice['finish_reason'] == 'max_output' or choice['finish_reason'] == 'length':
            finish_reason = 'length'
        if body.get('output_sensitive') or body.get('input_sensitive'):
            finish_reason = 'content_filter'
        r['choices'].append({
            'finish_reason': finish_reason,
            'index': choice['index'],
            'message': {
                'role': "assistant",
                'content': choice['text'],
            },
        })
    return r


def convert_sse_response(body, model, id=None):
    """Convert MiniMax SSE response to OpenAI format"""
    response =  {
        'id': id,
        'object': "chat.completion.chunk",
        'created': body['created'],
        'model': model,
        'choices': [
            {
                'index': 0,
                'delta': {},
                'finish_reason': None,
            }
        ],
    }

    if 'base_resp' in body:
        # stream end
        minimax_finish_reason = body['choices'][0]['finish_reason']
        finish_reason = 'stop'
        if minimax_finish_reason == 'max_output' or minimax_finish_reason == 'length':
            finish_reason = 'length'
        if body.get('output_sensitive') or body.get('input_sensitive'):
            finish_reason = 'content_filter'
        response['choices'][0]['finish_reason'] = finish_reason
        response['choices'][0]['delta'] = {
            'content': body['choices'][0]['delta'],
        }
    elif 'choices' not in body:
        response['choices'][0]['delta'] = {
            'role': "assistant",
            "content": "",
        }
    else:
        response['choices'][0]['delta'] = {
            'content': body['choices'][0]['delta'],
        }
    return json.dumps(response, ensure_ascii=False)
