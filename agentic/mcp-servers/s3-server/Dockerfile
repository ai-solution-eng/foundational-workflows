# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies to /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==================== Final Stage ====================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Copy application code
COPY s3_server.py .

# Create non-root user and data directory
RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app /opt/venv

# Switch to non-root user
USER mcpuser

# Environment variables with defaults
ENV PORT=9097 \
    HOST=0.0.0.0

# Expose port
EXPOSE 9097

# Run the MCP server
# FastMCP automatically uses uvicorn when running with streamable-http transport
CMD ["python", "-u", "s3_server.py"]