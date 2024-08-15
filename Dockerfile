# Use the official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libportaudio2 \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .
COPY src/chatbot.py src/
COPY src/css.py src/
COPY src/js.py src/

# Expose the correct port for Elastic Beanstalk
EXPOSE 8080

# Use shell form CMD to support $PORT (optional)
# BUT: better to hardcode to 8080 because EB expects it
CMD streamlit run src/chatbot.py --server.port=8080 --server.address=0.0.0.0 --server.enableCORS=false
