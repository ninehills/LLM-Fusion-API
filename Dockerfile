FROM python:3.11-slim-bullseye
LABEL maintainer="Tao Yang <swulling@gmail.com>"

WORKDIR /app
COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir /tmp/requirements.txt
COPY ./llm_fusion_api /app/llm_fusion_api

EXPOSE 8080

CMD ["uvicorn", "llm_fusion_api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]