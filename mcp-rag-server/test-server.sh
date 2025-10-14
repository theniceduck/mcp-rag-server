#!/bin/bash
# Test script to verify MCP server is working

echo "Testing MCP RAG Server..."
echo ""

echo "1. Checking if containers are running..."
docker ps | grep -E "(mcp-rag-server|mcp-ollama)"
echo ""

echo "2. Checking Ollama models..."
docker exec mcp-ollama ollama list
echo ""

echo "3. Checking server logs (last 20 lines)..."
docker logs --tail 20 mcp-rag-server
echo ""

echo "4. Checking data directories..."
docker exec mcp-rag-server ls -la /data/
echo ""

echo "Test complete!"
echo ""
echo "To test with a real PDF:"
echo "1. Place a PDF in ./shared/ directory"
echo "2. In Cursor AI, say: 'Upload /shared/your-file.pdf to the RAG server'"
echo "3. Then ask questions about it!"

