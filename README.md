# Chatbot RHM API

Medical chatbot API with RAG (Retrieval Augmented Generation) using Qdrant vector database.

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Make (optional, for using Makefile commands)

## Quick Start


### Option 1: Run Locally

```bash
# Copy local environment config
copy .env.local .env

# Install dependencies
pip install -r requirements.txt

# Start local Qdrant (if not running)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant


# Start API server
python start_api.py
```


### Option 2: Run with Docker 

```bash
# Copy docker environment config
copy .env.docker .env

# Build and start all services
docker-compose up --build -d

# Check status
docker-compose ps
```
### Important – Final step: Load data into Qdrant after building Docker

- API docs: `http://localhost:8000/api/docs`
- Endpoint: `POST /api/embeddings/load`
- Click execute , it will load all data by default.


## Environment Configuration

- `.env.local` - Local development (uses `localhost:6333` for Qdrant)
- `.env.docker` - Docker deployment (uses `qdrant:6333` service name)
