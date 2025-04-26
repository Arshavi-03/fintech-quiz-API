FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p static templates

# Use shell form to allow variable expansion
CMD uvicorn main:app --host 0.0.0.0 --port $PORT