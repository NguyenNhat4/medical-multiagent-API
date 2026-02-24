# Chatbot RHM API

This project is an intelligent Multi-Agent system designed to serve as a reliable medical information bridge for both healthcare professionals (facilitating cross-training between dentistry and endocrinology) and patients (providing accessible diabetes care advice). To guarantee strict medical accuracy and eliminate AI hallucinations, the system utilizes a Retrieval-Augmented Generation (RAG) architecture over a closed-domain dataset of approximately 3,100 expert-curated Q&A pairs, ensuring responses are retrieved from vetted knowledge rather than generated from scratch. Specialized agents dynamically adapt the retrieved information, delivering in-depth clinical terminology for doctors and clear, everyday language for patients. Future development will expand this architecture to dynamically query authoritative external medical databases, such as PubMed, for even broader, real-time information retrieval.

# System architecture diagram
![alt text](images/system-architecture.png)

# Med flows diagram
![alt text](images/medflow.png)

# User agentic memory diagram
![alt text](images/user-agentic-memory.png)

# Medical knowledge ingestion pipeline
![alt text](images/medical-knowledge-ingestion-pipeline.png)


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
# run this notebook to convert data to vector and ingrest to qdrant vector db
qdrant.ipynb

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
