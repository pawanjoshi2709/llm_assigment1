import json
import re
from pathlib import Path

import ollama
import chromadb

try:
    import pypdf;    HAS_PDF  = True
except ImportError:  HAS_PDF  = False
try:
    import docx;     HAS_DOCX = True
except ImportError:  HAS_DOCX = False

KB_FOLDER       = Path("kb")
CHROMA_DB_PATH  = Path("chroma_db")
COLLECTION_NAME = "hr_negotiation"
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "llama3.1:8b"
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 120
TOP_K           = 4

KB_FOLDER.mkdir(exist_ok=True)
CHROMA_DB_PATH.mkdir(exist_ok=True)

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client      = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        if not HAS_PDF:
            return ""
        reader = pypdf.PdfReader(str(path))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    elif ext in (".docx", ".doc"):
        if not HAS_DOCX:
            return ""
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    elif ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return "\n\n---\n\n".join(json.dumps(i, indent=2) for i in data)
        return json.dumps(data, indent=2)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start: start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    return [ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]


def ingest(clear_first=False) -> str:
    col = get_collection()
    log = []

    if clear_first:
        existing = col.get()
        if existing["ids"]:
            col.delete(ids=existing["ids"])
        log.append("Cleared existing vectors.")

    supported = {".txt", ".md", ".pdf", ".docx", ".doc", ".json", ".csv"}
    files = [f for f in KB_FOLDER.rglob("*") if f.suffix.lower() in supported]

    if not files:
        return f"No documents found in {KB_FOLDER}/"

    total = 0
    for file in files:
        try:
            text   = load_document(file)
            chunks = chunk_text(text)
            ids    = [f"{file.stem}_c{i}" for i in range(len(chunks))]
            metas  = [{"source": file.name, "chunk_index": i} for i in range(len(chunks))]

            existing_ids = set(col.get(ids=ids)["ids"])
            new_idx = [i for i, cid in enumerate(ids) if cid not in existing_ids]

            if new_idx:
                for start in range(0, len(new_idx), 10):
                    batch = new_idx[start: start + 10]
                    col.add(
                        documents =[chunks[i] for i in batch],
                        ids       =[ids[i]    for i in batch],
                        metadatas =[metas[i]  for i in batch],
                        embeddings=embed([chunks[i] for i in batch]),
                    )

            total += len(chunks)
            log.append(f"✅ {file.name} → {len(chunks)} chunks ({len(new_idx)} new)")
        except Exception as e:
            log.append(f"⚠️ {file.name} → ERROR: {e}")

    log.append(f"\nTotal chunks in DB: {total}")
    return "\n".join(log)


def retrieve(query: str, top_k=TOP_K) -> list[dict]:
    col   = get_collection()
    q_emb = embed([query])[0]
    res   = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {"text": doc, "source": meta.get("source", "?"), "score": round(1 - dist, 3)}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


SYSTEM_PROMPT = """You are an expert HR executive assistant specialising in salary negotiations.
Answer ONLY from the provided KB context. Be specific — quote techniques and example phrases.
Structure: Recommendation → Technique(s) → Example phrases → Caveats.
If context is insufficient, say so clearly."""

EXPANSION_PROMPT = """Given a salary negotiation question, return ONLY a JSON list of 3 strings:
the original query plus 2 alternative phrasings. No explanation."""


def expand_query(query: str) -> list[str]:
    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXPANSION_PROMPT},
                {"role": "user",   "content": query}
            ],
            options={"temperature": 0.4}
        )
        match = re.search(r'\[.*\]', resp["message"]["content"], re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return [query]


def ask(query: str, top_k=TOP_K, use_expansion=True) -> tuple[str, list[dict], list[str]]:
    queries = expand_query(query) if use_expansion else [query]

    seen, chunks = set(), []
    for q in queries:
        for chunk in retrieve(q, top_k):
            key = chunk["text"][:80]
            if key not in seen:
                seen.add(key)
                chunks.append(chunk)

    chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)[:top_k]

    if not chunks:
        return "No relevant documents found. Please run ingest first.", [], queries

    ctx = "\n\n".join(
        f"[{i+1}] {c['source']} (score={c['score']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    prompt = f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        options={"temperature": 0.3}
    )
    return response["message"]["content"], chunks, queries


def kb_info() -> str:
    files = [f for f in KB_FOLDER.rglob("*") if f.is_file()]
    lines = [f"📁 kb/ — {len(files)} file(s)"]
    for f in files:
        lines.append(f"  • {f.name} ({f.stat().st_size:,} bytes)")
    try:
        count = get_collection().count()
        lines.append(f"🗂️ ChromaDB chunks: {count}")
    except Exception as e:
        lines.append(f"ChromaDB: {e}")
    return "\n".join(lines)
