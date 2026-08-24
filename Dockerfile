FROM python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

ARG SENTINEX_EXTRAS=ui

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    SENTINEX_MODE=demo \
    SENTINEX_HOST=0.0.0.0

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY pyproject.toml README.md ./
COPY models ./models
COPY app.py ./
RUN python -m pip install --upgrade "pip==26.2.1" "setuptools==84.0.0" \
    && python -m pip install --no-cache-dir -e ".[${SENTINEX_EXTRAS}]"

USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3)"

CMD ["python", "app.py"]
