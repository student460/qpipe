FROM python:3.12-slim

WORKDIR /app

# Install the package (nautilus_trader ships manylinux wheels for amd64 + arm64)
COPY pyproject.toml README.md ./
COPY qpipe ./qpipe
RUN pip install --no-cache-dir .

# Default content baked in as fallbacks; real state comes from volumes (see compose)
COPY configs ./configs
COPY data ./data

EXPOSE 8000 8765

CMD ["qpipe", "serve", "--host", "0.0.0.0"]
