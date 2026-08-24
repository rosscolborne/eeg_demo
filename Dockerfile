FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install .

COPY brainflow_service/ /app/brainflow_service/

CMD ["uvicorn", "brainflow_service.app:app", "--host", "0.0.0.0", "--port", "10000"]
