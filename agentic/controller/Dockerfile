FROM python:3.11-slim

WORKDIR /app

# Copy operator code
COPY model-key-controller.py .
COPY requirements.txt .

# Install kubernetes client (compatible with k8s 1.32+)
RUN pip install --no-cache-dir -r requirements.txt

# Run as non-root user
RUN useradd -m -u 1000 model-key-controller && chown -R model-key-controller:model-key-controller /app
USER model-key-controller

# Run the controller
CMD ["python", "model-key-controller.py"]