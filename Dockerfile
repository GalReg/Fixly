# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (excluding vectorstore for now)
COPY . .

# Copy vectorstore data separately to preserve existing data
COPY vectorstore/ vectorstore/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (if needed for webhooks)
EXPOSE 8000

# Command to run the bot
CMD ["python", "bot_w_langchain_rag.py"]