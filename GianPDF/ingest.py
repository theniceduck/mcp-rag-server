# === ingest.py ===
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ---------- CONFIG ----------
PDF_PATH = "docs/DIAGNOSTIK DAN PENGURUSAN PEROSAK TANAMAN INDUSTRI.pdf"     # <- change this path when you switch PDF
CHROMA_DIR = "chroma_db"                # local Chroma folder
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "embeddinggemma:300m"
# ----------------------------

def main():
    # Derive a stable, unique collection name from the PDF filename
    collection_name = Path(PDF_PATH).stem.lower().replace(" ", "_")

    # 1) Load PDF
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No pages found in {PDF_PATH}")

    # 2) Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(docs)
    if not splits:
        raise RuntimeError("No chunks produced — check your PDF or splitter settings.")

    # 3) Embeddings (via Ollama)
    emb = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 4) Create / persist Chroma index
    # NOTE: When persist_directory is provided, persisting is automatic — no .persist() call.
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=emb,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
    )

    count = vectordb._collection.count()  # quick sanity check
    print(f"✅ Ingested {len(splits)} chunks into collection '{collection_name}' "
          f"(stored={count}) at '{CHROMA_DIR}'")

if __name__ == "__main__":
    main()

