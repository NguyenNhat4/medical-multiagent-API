
.PHONY: help local docker-build docker-up docker-down load-vectors clean

help:
	@echo "Available commands:"
	@echo "  make local         - Run locally (copy .env.local, load vectors, start API)"
	@echo "  make docker-build  - Build and run Docker (copy .env.docker, build, up, load vectors)"
	@echo "  make docker-up     - Start existing Docker containers (copy .env.docker, up, load vectors)"
	@echo "  make docker-down   - Stop Docker containers"
	@echo "  make load-vectors  - Load vectors to Qdrant only"
	@echo "  make clean         - Clean up containers and volumes"

# 1. Run LOCAL: copy .env.local → load vectors → start API
local:
	@echo "=== Running in LOCAL mode ==="
	copy .env.local .env
	@echo "Waiting for local Qdrant to be ready..."
	timeout /t 2 /nobreak >nul
	python loadvector_qdrant.py
	python start_api.py

# 2. Docker BUILD: copy .env.docker → build → up → wait → load vectors
docker-build:
	@echo "=== Building and running DOCKER containers ==="
	copy .env.docker .env
	docker-compose down
	docker-compose build
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	timeout /t 10 /nobreak >nul
	python loadvector_qdrant.py
	@echo "=== Docker containers are running ==="
	docker-compose ps

# 3. Docker UP: copy .env.docker → up → wait → load vectors
docker-up:
	@echo "=== Starting existing DOCKER containers ==="
	copy .env.docker .env
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	timeout /t 10 /nobreak >nul
	python loadvector_qdrant.py
	@echo "=== Docker containers are running ==="
	docker-compose ps

# Stop Docker containers
docker-down:
	@echo "=== Stopping Docker containers ==="
	docker-compose down

# Load vectors only (Qdrant must be running)
load-vectors:
	@echo "=== Loading vectors to Qdrant ==="
	python loadvector_qdrant.py

# Clean everything (containers + volumes)
clean:
	@echo "=== Cleaning up Docker containers and volumes ==="
	docker-compose down -v
	@echo "=== Cleanup complete ==="