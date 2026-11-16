# Shared base image for API and target.
FROM python:3.11-slim
# No .pyc files; unbuffered logs.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# All commands run from the app root.
WORKDIR /app
# Cache dependency installation separately.
COPY requirements.txt .
# Install pinned runtime requirements.
RUN pip install --no-cache-dir -r requirements.txt
# Application package.
COPY src/ ./src/
# Dataset and generated reference artifacts.
COPY data/ ./data/
# Writable location for the prediction DB.
RUN mkdir -p /app/logs
# Default API port.
EXPOSE 8000
# Default service command.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
