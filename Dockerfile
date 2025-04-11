FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables similar to Render
ENV PORT=10000
ENV PYTHONUNBUFFERED=1
ENV CORS_ALLOWED_ORIGINS="http://localhost:3000,http://localhost:3001,https://alu-student-companion.onrender.com"

# Start the application
CMD ["python", "server.py"]
