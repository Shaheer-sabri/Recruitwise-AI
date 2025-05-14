FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a .env file if it doesn't exist
RUN touch .env

# Expose the port the app runs on
EXPOSE 4002

# Command to run the application
CMD ["uvicorn", "scorer_api:app", "--host", "0.0.0.0", "--port", "4002"]