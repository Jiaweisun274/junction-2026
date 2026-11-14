# Agent container. Everything the harness needs is installed at build time, so at run time the network can be
# restricted to the tutor sidecar, model APIs and the npm registry (task.toml allowlist).
# Alpine/musl makes Harbor's OpenCode installer use the packaged Node instead of downloading nvm from GitHub.
FROM python:3.12-alpine

RUN apk add --no-cache bash curl coreutils nodejs npm ca-certificates git \
 && npm i -g opencode-ai@latest \
 && pip install --no-cache-dir "mcp[cli]==1.9.4" "numpy==2.2.6" "pydantic==2.11.10"

WORKDIR /app
