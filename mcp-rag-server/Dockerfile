FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

# Create data directory
RUN mkdir -p /data/chroma_db /data/uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHROMA_DIR=/data/chroma_db
ENV UPLOAD_DIR=/data/uploads
ENV OLLAMA_HOST=http://ollama:11434

# Make server executable
RUN chmod +x server.py

# Run the server
CMD ["python", "server.py"]

