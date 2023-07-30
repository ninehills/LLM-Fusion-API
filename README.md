# LLM-Fusion-API

## Supported API

- [OpenAI](https://platform.openai.com/docs/api-reference/introduction)
- [Wenxin](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/flfmc9do2)
- [FastChat](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md)
- [MiniMax](https://api.minimax.chat/)
- [Zhipu](https://open.bigmodel.cn/doc/api#overview)
    - Also known as ChatGLM.

## OpenAI API Compatibility

### Chat Completion

| API | system message | function | stream | temperature | top_p | n | stop | max_tokens | presence_penalty | frequency_penalty | logit_bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| Wenxin | ❌* |❌ | ✔️ | ✔️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| FastChat | ✔️ | ❌ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ❌ | ❌ | ❌ |
| MiniMax | ✔️ | ❌ | ✔️ | ✔️ | ✔️ | ✔️ | ❌ | ✔️ | ❌ | ❌ | ❌ |
| Zhipu | ❌ | ❌ | ✔️ | ✔️ | ✔️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

TODO:

1. support minimax function call
2. support embedding api
3. add a simple frontend

## Running the API

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# edit .env to set OPENAI_API_KEY etc.

# run the API
uvicorn llm_fusion_api:app --reload
```

### Test the API

```bash
$ curl localhost:8000/v1/models 2>/dev/null| jq ".data[].id"
"gpt-3.5-turbo-16k-0613"
"gpt-3.5-turbo-0301"
"gpt-3.5-turbo-16k"
"gpt-4-0613"
"gpt-4-0314"
"text-embedding-ada-002"
"gpt-4"
"gpt-3.5-turbo-0613"
"gpt-3.5-turbo"
"wenxin/ernie-bot"
"wenxin/ernie-bot-turbo"
"wenxin/bloomz_7b1"
"wenxin/embedding-v1"
"minimax/abab5.5-chat"
"minimax/embo-01"

$ curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer xxx" \
  -d '{ "stream": true,
    "model": "minimax/abab5.5-chat",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
data: {"id": "abe9437a622c413abd157605efb6e228", "object": "chat.completion.chunk", "created": 1690690250, "model": "abab5.5-chat", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}

data: {"id": "abe9437a622c413abd157605efb6e228", "object": "chat.completion.chunk", "created": 1690690250, "model": "abab5.5-chat", "choices": [{"index": 0, "delta": {"content": "Hello! How can I assist you today?"}, "finish_reason": null}]}

data: {"id": "abe9437a622c413abd157605efb6e228", "object": "chat.completion.chunk", "created": 1690690250, "model": "abab5.5-chat", "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}]}

data: [DONE]
```