FROM python:3.10-slim

ARG HTTP_PROXY=http://proxy.stadt.hamburg.de:3128
ARG HTTPS_PROXY=http://proxy.stadt.hamburg.de:3128
ARG NO_PROXY=localhost,127.0.0.1

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY whole_process_contributions_with_comments_250.parquet .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
