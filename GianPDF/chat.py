# === chat.py ===
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ---------- CONFIG ----------
PDF_PATH = "docs/DIAGNOSTIK DAN PENGURUSAN PEROSAK TANAMAN INDUSTRI.pdf"   # <- must match the PDF you ingested
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "embeddinggemma:300m"
LLM_MODEL = "deepseek-r1:7b"
TOP_K = 4
TEMPERATURE = 0.2
# ----------------------------

def build_chain():
    # The collection name must match ingest.py (derived from the same PDF path)
    collection_name = Path(PDF_PATH).stem.lower().replace(" ", "_")

    # 1) Vector DB + retriever
    emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
        embedding_function=emb,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    # 2) LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)

    # 3) Conversational memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # 4) Conversational RAG chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",              # simple & effective
        return_source_documents=True,
        verbose=False,
    )
    return qa

def pretty_sources(source_documents):
    if not source_documents:
        return
    print("\nSources:")
    for i, d in enumerate(source_documents, 1):
        meta = d.metadata or {}
        src = meta.get("source", "pdf")
        page = meta.get("page")
        try:
            page = int(page) + 1 if page is not None else "?"
        except Exception:
            page = "?"
        print(f"[{i}] {src} p.{page}")

def ask(qa, q: str):
    res = qa({"question": q})
    ans = res["answer"]
    print("\nUSER:", q)
    print("BOT :", ans.strip())
    pretty_sources(res.get("source_documents", []))

if __name__ == "__main__":
    qa = build_chain()
    # Sample turns — modify freely
    ask(qa, "Give me a concise summary of the main topic in this PDF.")


