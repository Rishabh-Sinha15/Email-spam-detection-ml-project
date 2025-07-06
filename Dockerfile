# Use official Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install basic system dependencies (for NLTK and pip)
RUN apt-get update && apt-get install -y \
    gcc \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy project files

COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (stopwords)
RUN python -m nltk.downloader -d /usr/share/nltk_data stopwords

# Set environment variable for NLTK to find downloaded data
ENV NLTK_DATA=/usr/share/nltk_data

# Run the Flask app
CMD ["python", "src/api.py"]
