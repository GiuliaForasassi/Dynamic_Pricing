FROM ghcr.io/astral-sh/uv:0.5.1-python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

ARG SRC_DIR
ARG AWS_REGION
ARG AWS_ACCOUNT_ID
ARG CODEARTIFACT_AUTH_TOKEN
ARG PYPI_INDEX_URL="https://aws:${CODEARTIFACT_AUTH_TOKEN}@mlcube-${AWS_ACCOUNT_ID}.d.codeartifact.${AWS_REGION}.amazonaws.com/pypi/ml3-platform/simple/"

# Run the following commands to install curl and build-essential (needed for c based packages)
# RUN apt-get update && apt-get install curl -y && apt-get install build-essential -y

# Install dependencies, excluding dev dependencies
# We pass the --frozen flag to ensure the lock file is NOT updated
# hence the same dependencies used in the development phase are used in production

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --all-extras --no-editable --index ${PYPI_INDEX_URL}

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching

ADD ${SRC_DIR} /app/${SRC_DIR}
ADD worker.py /app
ADD pyproject.toml /app
ADD uv.lock /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-extras --no-editable --index ${PYPI_INDEX_URL}


FROM python:3.12-slim-bookworm

ARG SECRET_SERVER_USERNAME
ARG SECRET_SERVER_PWD
ENV SECRET_SERVER_USERNAME=$SECRET_SERVER_USERNAME
ENV SECRET_SERVER_PWD=$SECRET_SERVER_PWD

COPY --from=builder --chown=app:app /app /app

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "worker.py"]