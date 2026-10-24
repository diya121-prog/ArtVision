# Multi-stage build: React Frontend + FastAPI PyTorch Backend
# Stage 1: Build the React Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY artvision/frontend/package*.json ./
RUN npm install
COPY artvision/frontend ./
RUN npm run build

# Stage 2: Python Machine Learning Backend
FROM python:3.11-slim
WORKDIR /app

# Install runtime system packages (OpenCV dependencies, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (saves 3+ GB by avoiding unnecessary CUDA drivers)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining backend dependencies
COPY artvision/requirements.txt ./
# Use opencv-python-headless for headless container environments
RUN sed -i 's/opencv-python/opencv-python-headless/' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and configurations
COPY artvision ./artvision
COPY showcase ./showcase
COPY predict.py ./

# Copy built frontend assets from Stage 1 into the backend directory
COPY --from=frontend-builder /app/frontend/dist /app/artvision/frontend/dist

# Set working directory to artvision
WORKDIR /app/artvision

# Expose port (7860 is standard for Hugging Face Spaces, 8000 for local)
EXPOSE 7860

# Run FastAPI server (uses $PORT if provided by host like Render/Railway, otherwise defaults to 7860)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
