docker run --name langfuse \
-e DATABASE_URL=postgresql://hiaivn:dev123@localhost:5433/chatbot_rhm \
-e NEXTAUTH_URL=http://localhost:3000 \
-e NEXTAUTH_SECRET=yLdi5cuIf8xHRqMgpAHkcHg+C08KK1PN/UZ+EuI00Ss \
-e SALT=mzoUWumriVXcjpx9M5kabs2tSAou6pWBNpqm3rlcsKs \
-e ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \ # generate via: openssl rand -hex 32
-p 3000:3000 \
-a STDOUT \
langfuse/langfuse