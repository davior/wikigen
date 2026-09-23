FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    PORT=5055

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

RUN useradd --create-home --uid 1000 wikigen \
    && mkdir -p /data \
    && chown -R wikigen:wikigen /data
USER wikigen

VOLUME ["/data"]
EXPOSE 5055

# Single worker: plans and SSE queues live in process memory.
CMD ["sh", "-c", "exec gunicorn -k gthread -w 1 --threads 8 --timeout 300 -b 0.0.0.0:${PORT} app:app"]
