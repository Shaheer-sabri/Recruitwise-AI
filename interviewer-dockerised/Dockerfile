FROM ubuntu:22.04

WORKDIR /home/app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python command (optional, for compatibility)
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code files
COPY . .

# Create directories for resumes and saved sessions
# RUN mkdir -p resumes interview_sessions

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]