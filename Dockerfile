# Use a slim Python image for smaller size
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies (including build tools for wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Ensure gunicorn is installed
    pip install gunicorn uvicorn && \
    apt-get remove -y gcc && \
    apt-get autoremove -y && \
    apt-get clean

# Copy your source code and model
COPY . .

# (Optional) Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port (Uvicorn will use this)
EXPOSE 8000

# Change working directory to src and run gunicorn from there
WORKDIR /app/src
CMD ["gunicorn", "predict_service:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]