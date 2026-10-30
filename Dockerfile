FROM python:3.13.16-slim-trixie AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get upgrade --yes \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /nonexistent --shell /usr/sbin/nologin app

COPY requirements-prod.txt pyproject.toml setup.py MANIFEST.in README.md ./
COPY src ./src
COPY schemas ./schemas
COPY benchmarks ./benchmarks
COPY examples ./examples
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini

RUN python -m pip install --upgrade pip setuptools==84.1.0 \
    && python -m pip install -r requirements-prod.txt \
    && python -m pip install --no-deps . \
    && python -m pip check \
    && python -c "from importlib.metadata import version; assert version('urllib3') == '2.8.0'; assert version('msgpack') == '1.2.3'; import fastapi, spacy, sqlalchemy, src.service.app_with_ui"

FROM base AS native
ENV NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN python -m pip install '.[pyreason]' \
    && python -m pip check \
    && python -m src.pyreason_qualification --iterations 4 --timeout-seconds 900 > /tmp/native-build-check.json \
    && python -c "import pyreason, pathlib; assert 'true' in (pathlib.Path(pyreason.__file__).parent / '.cache_status.yaml').read_text().lower()" \
    && chown -R 10001:10001 /usr/local/lib/python3.13/site-packages/pyreason/cache \
    && python -m pip uninstall --yes setuptools pip \
    && rm -rf /usr/local/lib/python3.13/ensurepip /root/.cache /tmp/native-build-check.json
ENV VLE_PYREASON_ENABLED=true VLE_PYREASON_WORKER_MODE=pooled VLE_PYREASON_WARMUP=true \
    NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2).read()" || exit 1

CMD ["uvicorn", "src.service.app_with_ui:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--timeout-keep-alive", "5", "--limit-concurrency", "200"]

FROM base AS runtime
RUN python -m pip uninstall --yes setuptools pip \
    && rm -rf /usr/local/lib/python3.13/ensurepip /root/.cache
USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2).read()" || exit 1

CMD ["uvicorn", "src.service.app_with_ui:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--timeout-keep-alive", "5", "--limit-concurrency", "200"]
