# MCP RAG Server for Jetson Orin Nano

A Model Context Protocol (MCP) server for interacting with PDF documents using RAG (Retrieval-Augmented Generation). Optimized for Jetson Orin Nano with existing Ollama installation.

## Prerequisites

- Jetson Orin Nano with JetPack
- Ollama installed and running with models:
  - `deepseek-r1:1.5b`
  - `embeddinggemma:300m`
- Docker installed
- Cursor AI (or any MCP-compatible client)

## Quick Start

### 1. Start the Server

```bash
cd mcp-rag-server

# Make scripts executable
chmod +x start-jetson.sh test-server.sh

# Start the server (automated with checks)
./start-jetson.sh

# Or start manually
docker compose up -d
```

### 2. Configure Cursor AI

```bash
# Copy configuration
cp mcp.json ~/.cursor/mcp.json
```

### 3. Restart Cursor AI

Close and reopen Cursor AI to load the MCP server.

## Usage

### Upload a Document

In Cursor AI chat:
```
Upload /path/to/document.pdf to the RAG server
```

Or place PDFs in the `shared/` folder:
```bash
cp ~/Documents/manual.pdf ./shared/
```
Then in Cursor:
```
Upload /shared/manual.pdf
```

Response will include a `session_id`.

### Query a Document

```
Using session [session_id], what are the main topics?
```

The server will return an answer with page citations.

### List All Sessions

```
List all RAG sessions
```

### Delete a Session

```
Delete session [session_id]
```

## Available MCP Tools

1. **upload_document** - Index a PDF for querying
   - Arguments: `file_path` (required), `session_id` (optional)
   - Returns: session_id, chunks_indexed, pages

2. **query_document** - Ask questions about indexed PDFs
   - Arguments: `session_id`, `question`, `top_k` (optional, default: 3), `include_sources` (optional)
   - Returns: answer, sources (with page numbers)

3. **list_sessions** - View all uploaded documents
   - Returns: List of all sessions with metadata

4. **get_session_info** - Get details about a session
   - Arguments: `session_id`
   - Returns: File info, chunk count, metadata

5. **delete_session** - Remove a document and its data
   - Arguments: `session_id`
   - Returns: Confirmation

## Configuration

Edit `docker-compose.yml` to adjust settings:

```yaml
environment:
  - OLLAMA_HOST=http://localhost:11434
  - EMBEDDING_MODEL=embeddinggemma:300m
  - LLM_MODEL=deepseek-r1:1.5b
  - CHUNK_SIZE=500              # Document chunk size
  - CHUNK_OVERLAP=100           # Overlap between chunks
  - TOP_K_DEFAULT=3             # Number of chunks to retrieve
  - TEMPERATURE=0.1             # LLM temperature (0.0-1.0)
  - MAX_CONTEXT_CHARS=4000      # Max context length
```

## Common Commands

```bash
# Start server
./start-jetson.sh

# Stop server
docker compose down

# View logs (real-time)
docker logs -f mcp-rag-server

# View last 50 lines of logs
docker logs --tail 50 mcp-rag-server

# Restart server
docker restart mcp-rag-server

# Monitor Jetson resources
sudo jtop

# Test server
./test-server.sh

# Check Ollama connection
curl http://localhost:11434/api/tags
```

## Logging

The server includes comprehensive logging that shows:

- **Startup configuration** - All environment variables and settings
- **Tool calls** - Which tools are being called and with what arguments
- **File operations** - PDF uploads, file copies, deletions
- **Processing steps** - PDF loading, chunking, embedding generation
- **Query execution** - Vector search, LLM generation, results
- **Errors and warnings** - Detailed error messages with context

### View Logs

```bash
# Follow logs in real-time (Ctrl+C to stop)
docker logs -f mcp-rag-server

# View recent logs
docker logs --tail 100 mcp-rag-server

# Search logs for specific session
docker logs mcp-rag-server 2>&1 | grep "session_id_here"

# View only errors
docker logs mcp-rag-server 2>&1 | grep ERROR
```

### Example Log Output

```
2025-01-15 10:23:45 [INFO] ============================================================
2025-01-15 10:23:45 [INFO] MCP RAG Server - Starting
2025-01-15 10:23:45 [INFO] ============================================================
2025-01-15 10:23:45 [INFO] OLLAMA_HOST: http://localhost:11434
2025-01-15 10:23:45 [INFO] EMBEDDING_MODEL: embeddinggemma:300m
2025-01-15 10:23:45 [INFO] LLM_MODEL: deepseek-r1:1.5b
2025-01-15 10:23:45 [INFO] CHUNK_SIZE: 500
2025-01-15 10:23:45 [INFO] TOP_K_DEFAULT: 3
2025-01-15 10:23:45 [INFO] Starting MCP server with stdio transport...
2025-01-15 10:23:45 [INFO] Server ready - waiting for client connections
2025-01-15 10:24:12 [INFO] Tool called: upload_document
2025-01-15 10:24:12 [INFO] Upload request for: /shared/manual.pdf
2025-01-15 10:24:12 [INFO] Session ID: abc123def456
2025-01-15 10:24:12 [INFO] Loading PDF...
2025-01-15 10:24:13 [INFO] PDF loaded: 45 pages
2025-01-15 10:24:13 [INFO] Splitting into chunks (size=500, overlap=100)...
2025-01-15 10:24:13 [INFO] Created 142 chunks
2025-01-15 10:24:13 [INFO] Generating embeddings using embeddinggemma:300m...
2025-01-15 10:24:25 [INFO] Indexing to ChromaDB collection: pdf_abc123def456
2025-01-15 10:24:26 [INFO] ✅ Upload complete: abc123def456 (142 chunks, 45 pages)
2025-01-15 10:25:03 [INFO] Tool called: query_document
2025-01-15 10:25:03 [INFO] Query request - Session: abc123def456, Question: What are the main topics?
2025-01-15 10:25:03 [INFO] Loading collection: pdf_abc123def456
2025-01-15 10:25:03 [INFO] Searching for top 3 relevant chunks...
2025-01-15 10:25:03 [INFO] Retrieved 3 chunks
2025-01-15 10:25:03 [INFO] Generating answer using deepseek-r1:1.5b...
2025-01-15 10:25:08 [INFO] Answer generated successfully
```

## Performance Optimization

### Enable Max Performance Mode

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Reduce Memory Usage

If experiencing memory issues, edit `docker-compose.yml`:

```yaml
environment:
  - TOP_K_DEFAULT=2             # Reduce from 3
  - CHUNK_SIZE=400              # Reduce from 500
  - TEMPERATURE=0.0             # More deterministic
```

Then restart:
```bash
docker compose up -d --force-recreate
```

## Troubleshooting

### Ollama Not Connecting

Ensure Ollama is listening on all interfaces:
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Models Not Found

Verify models are installed:
```bash
ollama list
# Should show: deepseek-r1:1.5b and embeddinggemma:300m
```

### Container Won't Start

Check logs:
```bash
docker logs mcp-rag-server
```

Rebuild:
```bash
docker compose up -d --build
```

### Out of Memory

1. Close other applications
2. Reduce chunk sizes and top_k (see Performance Optimization)
3. Monitor with: `sudo jtop`

### Connection Issues

Test from container:
```bash
docker exec -it mcp-rag-server bash
curl http://localhost:11434/api/tags
```

If fails, verify Ollama is accessible from host network.

## How It Works

```
┌─────────────────────────────────┐
│      Cursor AI (Your IDE)       │
└────────────────┬────────────────┘
                 │ MCP Protocol
┌────────────────▼────────────────┐
│   MCP RAG Server (Container)    │
│   • LangChain RAG Pipeline      │
│   • ChromaDB Vector Database    │
└────────────────┬────────────────┘
                 │ HTTP localhost:11434
┌────────────────▼────────────────┐
│   Your Ollama (Host System)     │
│   • deepseek-r1:1.5b (LLM)      │
│   • embeddinggemma:300m (Embed) │
└─────────────────────────────────┘
```

### Document Upload Flow

1. PDF uploaded via MCP tool
2. PyPDFLoader extracts text from pages
3. RecursiveCharacterTextSplitter chunks the text (500 chars, 100 overlap)
4. OllamaEmbeddings converts chunks to vectors (embeddinggemma:300m)
5. ChromaDB stores vectors + metadata (page numbers)

### Query Flow

1. User asks a question
2. Question converted to vector (embeddinggemma:300m)
3. ChromaDB performs similarity search (top 3 chunks)
4. Context built from retrieved chunks
5. LLM generates answer (deepseek-r1:1.5b)
6. Response returned with page citations

## File Locations

### On Host (Jetson)
- Project: `~/mcp-rag-server/`
- Shared PDFs: `~/mcp-rag-server/shared/`
- Cursor config: `~/.cursor/mcp.json`

### In Container
- Uploaded PDFs: `/data/uploads/`
- Vector DB: `/data/chroma_db/`
- Shared mount: `/shared/`

### Docker Volumes
- `mcp_data` - Persistent storage for PDFs and ChromaDB

## Architecture Details

### Optimizations for Jetson

- **Host networking**: Direct access to Ollama (no bridge overhead)
- **Reduced chunk size**: 500 chars (vs 800 standard) for 1.5b model
- **Fewer chunks**: top_k=3 (vs 4 standard) for faster processing
- **Smaller context**: 4000 chars (vs 8000) fits in 1.5b context window
- **No Ollama container**: Uses your existing installation

### Technology Stack

- **MCP Protocol**: Cursor AI integration
- **LangChain**: Document processing and RAG pipeline
- **ChromaDB**: Vector database for semantic search
- **Ollama**: Local LLM runtime (embeddings + generation)
- **Docker**: Containerized deployment
- **Python 3.11**: ARM64 native

## Security Notes

⚠️ **Development Setup** - Not production-ready:
- No authentication
- No rate limiting
- No input validation
- Local access only

For production use, implement:
- API authentication
- Rate limiting
- Input sanitization
- HTTPS/TLS
- User isolation
- Audit logging

## Stopping the Server

```bash
# Stop and keep data
docker compose down

# Stop and remove all data
docker compose down -v
```

## Credits

- Based on GianPDF implementation
- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Ollama](https://ollama.ai)
- Uses [ChromaDB](https://www.trychroma.com)
- Implements [Model Context Protocol](https://github.com/anthropics/model-context-protocol)

## License

Free to use and modify. No restrictions.
