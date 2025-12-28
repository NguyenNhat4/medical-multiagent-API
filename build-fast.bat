@echo off
echo ==========================================
echo Docker Build with Cache Optimization
echo ==========================================

REM Enable BuildKit for faster builds
set DOCKER_BUILDKIT=1
set COMPOSE_DOCKER_CLI_BUILD=1

echo.
echo [1/3] Building with Docker Compose...
docker-compose build --parallel

echo.
echo [2/3] Tagging latest image...
docker tag chatbot-rhm-api chatbot-rhm-api:latest

echo.
echo [3/3] Done! Starting services...
docker-compose up -d

echo.
echo ==========================================
echo Build completed successfully!
echo ==========================================
echo.
echo View logs: docker-compose logs -f chatbot-rhm-api
echo Stop services: docker-compose down
