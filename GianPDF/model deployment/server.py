# server.py (simple, explicit RAG — rock solid)
import os
import re
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ---------- CONFIG ----------
CHROMA_DIR = "chroma_db"
UPLOAD_DIR = "uploads"
EMBEDDING_MODEL = "embeddinggemma:300m"
LLM_MODEL = "deepseek-r1:7b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K_DEFAULT = 4
TEMPERATURE = 0.1   # lower = fewer hallucinations
MAX_CONTEXT_CHARS = 8000  # safety: clip stuffed prompt context
# ----------------------------

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="GianPDF")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    session_id: str
    top_k: int | None = None
    debug: bool | None = False  # optional: return a tiny context preview

def _session_collection(session_id: str) -> str:
    return f"pdf_{session_id}".lower()

def _strip_think(text: str) -> str:
    if not text:
        return text
    # Remove <think> blocks and related scaffolding
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"(?im)^\s*(thought|reasoning|deliberate|chain[- ]?of[- ]?thought)\s*:.*$", "", text)
    # Trim blank lines
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines).strip()

def _build_vectordb(collection_name: str, embedding_model: str = EMBEDDING_MODEL) -> Chroma:
    emb = OllamaEmbeddings(model=embedding_model)
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
        embedding_function=emb,
    )

def _stuff_context(docs: List, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Join top documents with page hints into a single context string, clipped to max_chars."""
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
    return f"""You are a precise assistant. Answer ONLY using the information in the CONTEXT.
If the answer is not present, say: "I don't know based on this PDF."
Be concise (2–15 sentences). Do not include any internal thoughts.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    session_id = uuid.uuid4().hex[:12]
    saved_path = Path(UPLOAD_DIR) / f"{session_id}.pdf"
    with saved_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load & chunk
    loader = PyPDFLoader(str(saved_path))
    docs = loader.load()
    if not docs:
        raise HTTPException(status_code=400, detail="No content found in the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not split the PDF into chunks.")

    # Index
    emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
    collection_name = _session_collection(session_id)
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
    )

    return {
        "ok": True,
        "session_id": session_id,
        "message": "PDF uploaded and indexed successfully.",
        "chunks_indexed": len(chunks),
    }

@app.post("/ask")
async def ask(req: AskRequest) -> Dict[str, Any]:
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is empty.")

    collection_name = _session_collection(req.session_id)

    # Ensure the collection exists
    try:
        vectordb = _build_vectordb(collection_name)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found. Upload a PDF first.")

    # Retrieve
    retriever = vectordb.as_retriever(search_kwargs={"k": (req.top_k or TOP_K_DEFAULT)})
    docs = retriever.get_relevant_documents(q)
    if not docs:
        return {
            "answer": "I couldn't find content related to that question in this PDF.",
            "sources": [],
            "debug": {"retrieved": 0} if req.debug else None,
        }

    # Build stuffed prompt with explicit context
    context_text = _stuff_context(docs)
    prompt = _rag_prompt(context_text, q)

    # Call the LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)
    result_msg = llm.invoke(prompt)  # returns a BaseMessage
    raw = getattr(result_msg, "content", "") if result_msg else ""
    answer = _strip_think(raw)

    # Build sources
    sources = []
    for d in docs:
        meta = d.metadata or {}
        page = meta.get("page")
        try:
            page = int(page) + 1 if page is not None else None
        except Exception:
            page = None
        sources.append({"source": meta.get("source", "pdf"), "page": page})

    out: Dict[str, Any] = {"answer": answer, "sources": sources}
    if req.debug:
        # Tiny preview to confirm it's reading your PDF
        out["debug"] = {
            "retrieved": len(docs),
            "context_preview": context_text[:400]
        }
    return out

