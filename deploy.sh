#!/bin/bash
# Deploy script for Render.com

echo "=== Building API ==="
docker build -t chat-docs-api .

echo "=== Building Frontend ==="
cd frontend
docker build -t chat-docs-web .

echo "=== Done! ==="
echo "To run locally: docker-compose -f docker-compose.full.yml up"
echo "To deploy to Render: render deploy"