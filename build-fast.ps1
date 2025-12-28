
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1

docker-compose build --parallel

docker tag chatbot-rhm-api chatbot-rhm-api:latest

docker-compose up -d
