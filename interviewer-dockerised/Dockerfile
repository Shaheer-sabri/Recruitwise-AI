FROM python:3.10-slim

WORKDIR /home/app

# Install system dependencies
#RUN apt-get update && apt-get install -y \
 #   build-essential \
  #  python3-dev \
   # poppler-utils \
    #&& rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code files
COPY . .

# Create directories for resumes and saved sessions
# RUN mkdir -p resumes interview_sessions

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]