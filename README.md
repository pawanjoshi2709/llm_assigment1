# 🤝 HR Salary Negotiation RAG

**Assignment 1 — Architecting and Deploying LLMs**
**Student:** Pawan Joshi | **GitHub:** https://github.com/pawanjoshi2709/llm_assigment1

---

## What This Does

A RAG-based AI assistant for HR executives doing salary negotiations.
Ask it anything about negotiation strategy — it retrieves relevant expert transcripts from the knowledge base and answers using `llama3.1:8b` running locally via Ollama.

---

## Quick Start

**Step 1 — Install dependencies**
```bash
pip install gradio ollama chromadb pypdf python-docx
```

**Step 2 — Pull Ollama models** (one-time, ~5 GB)
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

**Step 3 — Start Ollama** (keep this terminal open)
```bash
ollama serve
```

**Step 4 — Ingest the knowledge base**
```bash
python -c "from rag import ingest; print(ingest())"
```

**Step 5 — Launch the UI**
```bash
python main.py
```
Opens at **http://localhost:7860**

---

## Project Structure

```
llm_assigment1/
├── rag.py                     # RAG engine — ingest, embed, retrieve, ask
├── main.py                    # Gradio UI — Chat + Knowledge Base tabs
├── requirements.txt
└── kb/
    ├── hr_negotiation.txt     # 8 expert negotiation transcripts (T001–T008)
    └── hr_negotiation2.txt    # 30 advanced Q&A pairs
```

---

## How RAG Works Here

```
Your Question
     │
     ▼
Query Expansion  →  llama3.1:8b generates 3 query variants
     │
     ▼
Embedding  →  nomic-embed-text converts each query to vectors
     │
     ▼
ChromaDB  →  cosine similarity search → top-4 unique chunks
     │
     ▼
Prompt  →  system prompt + retrieved context + your question
     │
     ▼
llama3.1:8b  →  answer with source attribution
```

---

## Knowledge Base

| File | Contents |
|---|---|
| `hr_negotiation.txt` | 8 full expert transcripts — accepted & declined outcomes, roles from SWE to Legal Counsel |
| `hr_negotiation2.txt` | 30 Q&A pairs — competing offers, pay cuts, equity alternatives, EMI constraints, top HR mistakes |

**Adding your own documents:** Drop any `.txt`, `.pdf`, `.docx`, or `.json` into `kb/` then use the **Knowledge Base** tab in the UI to re-ingest.

---

## Sample Queries to Try

```
Is it possible to convince a recruit to accept 10% less static pay? Show me how.
How do I handle a candidate with a competing offer 15% higher than ours?
What non-monetary benefits should I highlight when cash budget is tight?
How do I reduce someone's salary during restructuring without losing them?
When does a joining bonus work and when does it NOT work?
The candidate is anchoring very high. How do I reframe the negotiation?
My recruit has high EMIs. Can I still close a salary gap of ₹2L?
```

---

## Stack

| Component | Technology |
|---|---|
| LLM | llama3.1:8b via Ollama |
| Embeddings | nomic-embed-text via Ollama (768-dim) |
| Vector DB | ChromaDB (local, persistent) |
| RAG Enhancement | Query expansion + multi-query deduplication |
| Frontend | Gradio |
| Supported file types | .txt .md .pdf .docx .json .csv |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` | Run `ollama serve` in a terminal |
| `Model not found` | Run `ollama pull llama3.1:8b` |
| Empty answers | Go to Knowledge Base tab → click **Ingest KB** |
| Slow responses (20s+) | Normal on CPU — GPU reduces to 2–5s |
