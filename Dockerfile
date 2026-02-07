FROM python:3.12-slim

WORKDIR /app

# Install git for the git tool
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

# Default workspace inside container
RUN mkdir /workspace

ENTRYPOINT ["agenticflow"]
CMD ["--help"]
