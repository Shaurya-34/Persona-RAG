FROM python:3.10-slim

# Install system dependencies (libgomp1 is required for Faiss)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
# Note: Since we are including the data_cache/ folder in our repo, 
# the app will be able to start immediately without running the pipeline.
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Prevent OpenMP runtime crashes
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
