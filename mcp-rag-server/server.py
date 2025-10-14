#!/usr/bin/env python3
"""
MCP RAG Server - Model Context Protocol server for RAG document interaction
Based on GianPDF implementation
"""
import os
import re
import uuid
import shutil
import json
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Sequence
from contextlib import asynccontextmanager
from datetime import datetime

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.stdio import stdio_server

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stderr)  # MCP uses stdout for protocol, so log to stderr
    ]
)
logger = logging.getLogger(__name__)
# ----------------------------

# ---------- CONFIG ----------
CHROMA_DIR = os.getenv("CHROMA_DIR", "/data/chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/data/uploads")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:1.5b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
# ----------------------------

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Set Ollama host
os.environ["OLLAMA_HOST"] = OLLAMA_HOST

# Log configuration
logger.info("=" * 60)
logger.info("MCP RAG Server - Starting")
logger.info("=" * 60)
logger.info(f"OLLAMA_HOST: {OLLAMA_HOST}")
logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"LLM_MODEL: {LLM_MODEL}")
logger.info(f"CHUNK_SIZE: {CHUNK_SIZE}")
logger.info(f"CHUNK_OVERLAP: {CHUNK_OVERLAP}")
logger.info(f"TOP_K_DEFAULT: {TOP_K_DEFAULT}")
logger.info(f"TEMPERATURE: {TEMPERATURE}")
logger.info(f"CHROMA_DIR: {CHROMA_DIR}")
logger.info(f"UPLOAD_DIR: {UPLOAD_DIR}")
logger.info("=" * 60)

# Initialize MCP server
app = Server("rag-server")
logger.info("MCP Server initialized")

def _session_collection(session_id: str) -> str:
    """Generate collection name from session ID."""
    return f"pdf_{session_id}".lower()

def _strip_think(text: str) -> str:
    """Remove <think> blocks from LLM responses."""
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"(?im)^\s*(thought|reasoning|deliberate|chain[- ]?of[- ]?thought)\s*:.*$", "", text)
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines).strip()

def _build_vectordb(collection_name: str) -> Chroma:
    """Build Chroma vector database instance."""
    emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
        embedding_function=emb,
    )

def _stuff_context(docs: list, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Join top documents into a single context string."""
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        page = meta.get("page")
        try:
            page = int(page) + 1 if page is not None else "?"
        except Exception:
            page = "?"
        snippet = d.page_content.strip()
        parts.append(f"[{i}] (p.{page}) {snippet}")
    ctx = "\n\n".join(parts)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n\n[...context truncated...]"
    return ctx

def _rag_prompt(context: str, question: str) -> str:
    """Generate RAG prompt."""
    return f"""You are a precise assistant. Answer ONLY using the information in the CONTEXT.
If the answer is not present, say: "I don't know based on this PDF."
Be concise (2–15 sentences). Do not include any internal thoughts.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    logger.info("Client requested tool list")
    return [
        Tool(
            name="upload_document",
            description="Upload and index a PDF document for RAG queries. Returns a session_id for subsequent queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PDF file to upload and index"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional custom session ID (auto-generated if not provided)"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="query_document",
            description="Ask a question about an uploaded PDF document using RAG",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID from upload_document"
                    },
                    "question": {
                        "type": "string",
                        "description": "Question to ask about the document"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of document chunks to retrieve (default: 4)",
                        "default": TOP_K_DEFAULT
                    },
                    "include_sources": {
                        "type": "boolean",
                        "description": "Include source page references in response (default: true)",
                        "default": True
                    }
                },
                "required": ["session_id", "question"]
            }
        ),
        Tool(
            name="list_sessions",
            description="List all available document sessions",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="delete_session",
            description="Delete a document session and its indexed data",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to delete"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="get_session_info",
            description="Get information about a specific session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to get information about"
                    }
                },
                "required": ["session_id"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    logger.info(f"Tool called: {name}")
    logger.debug(f"Arguments: {arguments}")
    
    if name == "upload_document":
        file_path = arguments.get("file_path")
        custom_session_id = arguments.get("session_id")
        
        logger.info(f"Upload request for: {file_path}")
        
        if not file_path:
            logger.error("No file_path provided")
            return [TextContent(type="text", text=json.dumps({"error": "file_path is required"}))]
        
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return [TextContent(type="text", text=json.dumps({"error": f"File not found: {file_path}"}))]
        
        if not str(file_path).lower().endswith(".pdf"):
            logger.error(f"Invalid file type: {file_path}")
            return [TextContent(type="text", text=json.dumps({"error": "Only PDF files are supported"}))]
        
        # Generate or use custom session ID
        session_id = custom_session_id if custom_session_id else uuid.uuid4().hex[:12]
        saved_path = Path(UPLOAD_DIR) / f"{session_id}.pdf"
        
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Copying file to: {saved_path}")
        
        # Copy file to upload directory
        shutil.copy2(file_path, saved_path)
        
        # Load & chunk
        logger.info("Loading PDF...")
        loader = PyPDFLoader(str(saved_path))
        docs = loader.load()
        if not docs:
            logger.error("No content found in PDF")
            return [TextContent(type="text", text=json.dumps({"error": "No content found in the PDF"}))]
        
        logger.info(f"PDF loaded: {len(docs)} pages")
        logger.info(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            logger.error("No chunks produced from PDF")
            return [TextContent(type="text", text=json.dumps({"error": "Could not split the PDF into chunks"}))]
        
        logger.info(f"Created {len(chunks)} chunks")
        logger.info(f"Generating embeddings using {EMBEDDING_MODEL}...")
        
        # Index
        emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
        collection_name = _session_collection(session_id)
        
        logger.info(f"Indexing to ChromaDB collection: {collection_name}")
        _ = Chroma.from_documents(
            documents=chunks,
            embedding=emb,
            persist_directory=CHROMA_DIR,
            collection_name=collection_name,
        )
        
        logger.info(f"✅ Upload complete: {session_id} ({len(chunks)} chunks, {len(docs)} pages)")
        
        result = {
            "success": True,
            "session_id": session_id,
            "message": "PDF uploaded and indexed successfully",
            "chunks_indexed": len(chunks),
            "pages": len(docs)
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "query_document":
        session_id = arguments.get("session_id")
        question = arguments.get("question", "").strip()
        top_k = arguments.get("top_k", TOP_K_DEFAULT)
        include_sources = arguments.get("include_sources", True)
        
        logger.info(f"Query request - Session: {session_id}, Question: {question[:100]}...")
        
        if not session_id:
            logger.error("No session_id provided")
            return [TextContent(type="text", text=json.dumps({"error": "session_id is required"}))]
        
        if not question:
            logger.error("No question provided")
            return [TextContent(type="text", text=json.dumps({"error": "question is required"}))]
        
        collection_name = _session_collection(session_id)
        
        # Ensure the collection exists
        logger.info(f"Loading collection: {collection_name}")
        try:
            vectordb = _build_vectordb(collection_name)
        except Exception as e:
            logger.error(f"Collection not found: {collection_name} - {e}")
            return [TextContent(type="text", text=json.dumps({
                "error": f"Session not found: {session_id}. Upload a PDF first.",
                "details": str(e)
            }))]
        
        # Retrieve
        logger.info(f"Searching for top {top_k} relevant chunks...")
        retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            logger.warning("No relevant documents found")
            return [TextContent(type="text", text=json.dumps({
                "answer": "I couldn't find content related to that question in this PDF.",
                "sources": []
            }))]
        
        logger.info(f"Retrieved {len(docs)} chunks")
        
        # Build stuffed prompt with explicit context
        context_text = _stuff_context(docs)
        prompt = _rag_prompt(context_text, question)
        
        logger.info(f"Generating answer using {LLM_MODEL}...")
        
        # Call the LLM
        llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)
        result_msg = llm.invoke(prompt)
        raw = getattr(result_msg, "content", "") if result_msg else ""
        answer = _strip_think(raw)
        
        logger.info("Answer generated successfully")
        
        # Build result
        result = {"answer": answer}
        
        if include_sources:
            sources = []
            for d in docs:
                meta = d.metadata or {}
                page = meta.get("page")
                try:
                    page = int(page) + 1 if page is not None else None
                except Exception:
                    page = None
                sources.append({"source": meta.get("source", "pdf"), "page": page})
            result["sources"] = sources
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "list_sessions":
        logger.info("Listing all sessions")
        sessions = []
        upload_dir = Path(UPLOAD_DIR)
        
        if upload_dir.exists():
            for pdf_file in upload_dir.glob("*.pdf"):
                session_id = pdf_file.stem
                file_size = pdf_file.stat().st_size
                created_time = pdf_file.stat().st_ctime
                
                sessions.append({
                    "session_id": session_id,
                    "file_name": pdf_file.name,
                    "file_size_bytes": file_size,
                    "created_timestamp": created_time
                })
        
        logger.info(f"Found {len(sessions)} sessions")
        result = {
            "sessions": sessions,
            "count": len(sessions)
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "delete_session":
        session_id = arguments.get("session_id")
        logger.info(f"Delete request for session: {session_id}")
        
        if not session_id:
            return [TextContent(type="text", text=json.dumps({"error": "session_id is required"}))]
        
        collection_name = _session_collection(session_id)
        pdf_path = Path(UPLOAD_DIR) / f"{session_id}.pdf"
        
        deleted = []
        
        # Delete PDF file
        if pdf_path.exists():
            logger.info(f"Deleting PDF file: {pdf_path}")
            pdf_path.unlink()
            deleted.append("pdf_file")
        
        # Delete Chroma collection
        try:
            logger.info(f"Deleting ChromaDB collection: {collection_name}")
            emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
            client = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=emb,
            )._client
            client.delete_collection(collection_name)
            deleted.append("vector_index")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")
        
        if deleted:
            logger.info(f"✅ Session deleted: {session_id} ({', '.join(deleted)})")
            result = {
                "success": True,
                "session_id": session_id,
                "deleted_components": deleted
            }
        else:
            logger.warning(f"Session not found: {session_id}")
            result = {
                "success": False,
                "error": f"Session not found: {session_id}"
            }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "get_session_info":
        session_id = arguments.get("session_id")
        logger.info(f"Info request for session: {session_id}")
        
        if not session_id:
            return [TextContent(type="text", text=json.dumps({"error": "session_id is required"}))]
        
        pdf_path = Path(UPLOAD_DIR) / f"{session_id}.pdf"
        
        if not pdf_path.exists():
            return [TextContent(type="text", text=json.dumps({
                "error": f"Session not found: {session_id}"
            }))]
        
        collection_name = _session_collection(session_id)
        
        # Get chunk count from Chroma
        try:
            vectordb = _build_vectordb(collection_name)
            chunk_count = vectordb._collection.count()
        except Exception:
            chunk_count = None
        
        result = {
            "session_id": session_id,
            "file_name": pdf_path.name,
            "file_size_bytes": pdf_path.stat().st_size,
            "created_timestamp": pdf_path.stat().st_ctime,
            "chunks_indexed": chunk_count,
            "collection_name": collection_name
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

async def main():
    """Run the MCP server."""
    logger.info("Starting MCP server with stdio transport...")
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Server ready - waiting for client connections")
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

