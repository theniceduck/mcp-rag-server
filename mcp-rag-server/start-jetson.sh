#!/bin/bash
# Quick start script for Jetson Orin Nano

set -e

echo "🚀 Starting MCP RAG Server on Jetson Orin Nano..."
echo ""

# Check if Ollama is running
echo "📡 Checking Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not accessible at localhost:11434"
    echo "   Please start Ollama first:"
    echo "   OLLAMA_HOST=0.0.0.0:11434 ollama serve"
    exit 1
fi

# Check for required models
echo ""
echo "🔍 Checking for required models..."
MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4)

if echo "$MODELS" | grep -q "embeddinggemma:300m"; then
    echo "✅ embeddinggemma:300m found"
else
    echo "⚠️  embeddinggemma:300m not found"
    echo "   Run: ollama pull embeddinggemma:300m"
fi

if echo "$MODELS" | grep -q "deepseek-r1:1.5b"; then
    echo "✅ deepseek-r1:1.5b found"
else
    echo "⚠️  deepseek-r1:1.5b not found"
    echo "   Run: ollama pull deepseek-r1:1.5b"
fi

# Create shared directory if it doesn't exist
echo ""
echo "📁 Setting up directories..."
mkdir -p ./shared
echo "✅ Shared directory ready at ./shared/"

# Start the container
echo ""
echo "🐳 Starting Docker container..."
docker compose up -d --build

# Wait for container to be ready
echo ""
echo "⏳ Waiting for server to start..."
sleep 3

# Check container status
if docker ps | grep -q mcp-rag-server; then
    echo "✅ MCP RAG Server is running!"
    echo ""
    echo "📊 Container status:"
    docker ps | grep mcp-rag-server
    echo ""
    echo "📝 View logs with:"
    echo "   docker logs -f mcp-rag-server"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Copy PDFs to ./shared/ directory"
    echo "   2. Configure Cursor AI (see README.md)"
    echo "   3. Start uploading documents!"
    echo ""
    echo "🛑 To stop:"
    echo "   docker compose down"
else
    echo "❌ Failed to start container"
    echo "   Check logs: docker logs mcp-rag-server"
    exit 1
fi

