# Python 3.12 slim image
FROM python:3.12-slim

WORKDIR /workspaces/rollerCoaster

# Copy requirements and Makefile first (for caching)
COPY requirements.txt .
COPY Makefile .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    make \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN make install

# Copy code and config
COPY scripts/ scripts/
COPY tests/ tests/
COPY .flake8 .flake8

ENV PYTHONPATH=/workspaces/rollerCoaster

USER root
CMD ["bash"]
