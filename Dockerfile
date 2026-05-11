# CREAMpy — containerised Python environment
# Zero Excel dependency. No GUI, no macro runtime, no Windows requirement.
#
# Build:
#   docker build -t creampy .
#
# Run validation (smoke-test on startup):
#   docker run --rm creampy
#
# Run full pipeline example:
#   docker run --rm creampy python examples/full_pipeline.py
#
# Run a single-shot welfare calculation and write JSON:
#   docker run --rm -v "$(pwd)/output:/out" creampy \
#       python -m creampy --yield-gain 0.15 --price 200 --qty 1000000 \
#       --out-json /out/result.json
#
# Interactive shell:
#   docker run --rm -it creampy bash

FROM python:3.12-slim

LABEL org.opencontainers.image.title="CREAMpy"
LABEL org.opencontainers.image.description="Bass diffusion + DREAM Closed Economy welfare model"
LABEL org.opencontainers.image.source="https://github.com/H1nokam1Kagura/CREAMpy"
LABEL org.opencontainers.image.licenses="MIT"

# No system packages needed — pure Python, zero runtime deps
WORKDIR /app

# Copy package source
COPY src/      src/
COPY tests/    tests/
COPY examples/ examples/
COPY eval/     eval/
COPY pyproject.toml .
COPY README.md .

# Install package + test runner (no openpyxl — eval mode only if user adds it)
RUN pip install --no-cache-dir -e ".[dev]"

# Default: run validation to confirm the image is healthy
CMD ["python", "-m", "creampy", "--validate"]
