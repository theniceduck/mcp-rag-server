#!/bin/bash
# Setup script for MCP Serverless RAG Server

set -e

echo "╔═══════════════════════════════════════════════╗"
echo "║   MCP Serverless RAG Server - Setup          ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# Check Python
echo "1️⃣  Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Python not found. Please install Python 3.8+."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✅ Python $PYTHON_VERSION found"

# Check Docker
echo ""
echo "2️⃣  Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

if ! docker ps &> /dev/null; then
    echo "❌ Docker daemon not running. Please start Docker."
    exit 1
fi

echo "✅ Docker is running"

# Check Ollama
echo ""
echo "3️⃣  Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running on localhost:11434"
else
    echo "⚠️  Ollama not accessible at localhost:11434"
    echo "   Please start Ollama:"
    echo "   OLLAMA_HOST=0.0.0.0:11434 ollama serve"
fi

# Check models
echo ""
echo "4️⃣  Checking Ollama models..."
MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4 || echo "")

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

# Install Python dependencies
echo ""
echo "5️⃣  Installing Python dependencies..."
$PYTHON_CMD -m pip install -q --upgrade pip
$PYTHON_CMD -m pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Create directories
echo ""
echo "6️⃣  Creating directories..."
mkdir -p uploads
mkdir -p ../GianPDF/chroma_db
echo "✅ Directories created"

# Build Docker image
echo ""
echo "7️⃣  Building Docker image..."
if docker images | grep -q "mcp-rag-server.*latest"; then
    echo "✅ Image mcp-rag-server:latest exists"
    echo "   (Skip rebuild. To rebuild: cd ../mcp-rag-server && docker build -t mcp-rag-server:latest .)"
else
    echo "Building from ../mcp-rag-server..."
    cd ../mcp-rag-server
    docker build -t mcp-rag-server:latest .
    cd ../mcp-serverless
    echo "✅ Image built successfully"
fi

# Summary
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║   ✅ Setup Complete!                          ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Test the launcher:"
echo "   $PYTHON_CMD test_serverless.py"
echo ""
echo "2. Configure Cursor:"
echo "   Add mcp.json config to Cursor's MCP settings"
echo "   See README.md for detailed instructions"
echo ""
echo "3. Start using in Cursor:"
echo "   The launcher will start automatically when LLM calls RAG tools"
echo ""
echo "📚 Documentation: See README.md"

