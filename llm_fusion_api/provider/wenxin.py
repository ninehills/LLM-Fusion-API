import json
import time
import httpx
import logging

from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from sse_starlette.sse import EventSourceResponse

from llm_fusion_api.provider.base import ChatHandler
from llm_fusion_api.response import ErrorResponse


logger = logging.getLogger(__name__)

class Wenxin(ChatHandler):
    cached_token: str = ""
    cached_token_expires_at: int = 0

    def __init__(self, wenxin_api_key: str, wenxin_secret_key: str):
        self.wenxin_api_key = wenxin_api_key
        self.wenxin_secret_key = wenxin_secret_key

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

    async def chat_completions(self, request: Request) -> Response:
        """https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11
        """
        token = await self.get_token()
        body = await request.json()
        logger.info(f"Wenxin request: {body}")
        new_body = convert_request(body)

        model = body.get('model', '').split('/')[-1]
        if model == 'ernie-bot':
            endpoint = "completions"
        elif model == 'ernie-bot-turbo':
            endpoint = "eb-instant"
        else:
            endpoint = model
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
            return JSONResponse(convert_response(res_body, model))

        # stream mode
        async def stream_generator():
            client = httpx.AsyncClient()
            first = True
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
                        yield convert_sse_response(first_payload, model)
                    yield convert_sse_response(payload, model)
            yield "[DONE]"

        return EventSourceResponse(stream_generator())


def convert_request(body):
    """Convert OpenAI request body to Wenxin format"""
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
        'stream': body.get('stream', False),
        'messages': msg,
    }
    if "temperature" in body:
        # Wenxin temperature is between 0.001 and 1
        new["temperature"] = min(max(body["temperature"], 0.001), 1)

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

def convert_sse_response(body, model):
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
        ],
    }

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
    return json.dumps(response, ensure_ascii=False)
