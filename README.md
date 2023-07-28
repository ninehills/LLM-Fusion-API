# LLM-Fusion-API

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
