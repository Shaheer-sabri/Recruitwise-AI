# Use a Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install compatible versions of packages
RUN pip install --no-cache-dir torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir huggingface_hub==0.12.1 \
    && pip install --no-cache-dir transformers==4.26.0 \
    && pip install --no-cache-dir sentence-transformers==2.2.2 \
    && pip install --no-cache-dir fastapi==0.103.1 uvicorn==0.23.2 python-multipart==0.0.6 \
    && pip install --no-cache-dir pdfplumber==0.10.1 nltk==3.8.1 numpy==1.25.2

# Download NLTK data during build to avoid runtime downloads
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY . .

# Expose port for the FastAPI application
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "parserapi:app", "--host", "0.0.0.0", "--port", "8000"]