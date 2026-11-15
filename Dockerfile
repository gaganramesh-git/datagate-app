# DataGate service image.
#
# NOTE: written without a local Docker daemon — build it once and check it before
# relying on it:  docker build -t datagate . && docker run -p 8000:8000 \
#                   -e DATAGATE_API_KEYS=change-me -v datagate-data:/data datagate
FROM python:3.12-slim

# DATAGATE_REQUIRE_LOGIN=1: the image is for hosting, so accounts are on by
# default. Unset it only for a private, single-user container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATAGATE_HOME=/data \
    DATAGATE_REQUIRE_LOGIN=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[service]"

# Run as an unprivileged user; all state lives in the /data volume.
RUN useradd --create-home --uid 10001 datagate && mkdir -p /data && chown datagate /data
USER datagate
VOLUME ["/data"]
EXPOSE 8000

# The service listens on all interfaces inside the container; with accounts on
# (above) every route needs a signed-in user.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"
CMD ["datagate", "serve", "--host", "0.0.0.0", "--port", "8000"]
