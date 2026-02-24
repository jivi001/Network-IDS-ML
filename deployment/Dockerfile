FROM python:3.11-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
COPY requirements/base.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install --prefix=/install -r /tmp/requirements.txt

FROM python:3.11-slim
WORKDIR /app
RUN useradd --create-home --shell /bin/bash appuser
COPY --from=builder /install /usr/local
COPY src ./src
COPY configs ./configs
ENV PYTHONPATH=/app
ENV MODEL_ARTIFACT_DIR=/app/experiments/latest
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=2).raise_for_status()"
USER appuser
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
