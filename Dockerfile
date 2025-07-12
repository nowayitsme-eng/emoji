# Use Python 3.10 (TensorFlow-compatible)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (OpenCV needs these)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port (Flask default)
EXPOSE 5000

# Run Gunicorn (Flask production server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
