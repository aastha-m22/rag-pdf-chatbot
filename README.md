#  RAG PDF Chatbot

Ask questions about any PDF document using Retrieval-Augmented Generation.  
Powered by **Groq** (LLaMA 3.3 70B), **LangChain**, **FAISS**, and **sentence-transformers**.

---

##  Features

- Upload any text-based PDF and index it in seconds
- Ask natural language questions — answers are grounded in the document
- Source attribution: every answer shows which chunks were used
- Three-file clean architecture: `config.py` / `rag_pipeline.py` / `app.py`
- All model names defined once in `config.py` — never hardcoded

---

##  How to Run (step-by-step)

### Step 1 — Create a virtual environment

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **CPU-only PyTorch** (saves ~2GB download — use if you don't have a GPU):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### Step 3 — Set your Groq API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

Get a free key at **[console.groq.com](https://console.groq.com)**.

### Step 4 — Run the app

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

##  Project Structure

```
rag-pdf-chatbot/
├── app.py              # Streamlit UI — sidebar, chat display, input handling
├── rag_pipeline.py     # PDF → chunks → FAISS → LLM → answer (no UI code)
├── config.py           # All settings in one place (model, chunk size, etc.)
├── .env                # Your API key (gitignored — never committed)
├── .env.example        # Template — copy and fill in
├── requirements.txt    # Pinned dependencies
└── README.md
```

---

##  Architecture

```
User types question
        │
        ▼
┌──────────────────────────────┐
│  app.py  (Streamlit UI)      │
│  · File upload               │
│  · Chat display              │
│  · Error messages            │
└──────────┬───────────────────┘
           │ calls
           ▼
┌──────────────────────────────┐
│  rag_pipeline.py             │
│                              │
│  process_pdf()               │
│    PyPDFLoader               │
│    → RecursiveTextSplitter   │
│    → HuggingFaceEmbeddings   │
│    → FAISS.from_documents()  │
│                              │
│  ask()                       │
│    FAISS similarity search   │
│    → ChatPromptTemplate      │
│    → ChatGroq (LLM)         │
│    → StrOutputParser         │
└──────────┬───────────────────┘
           │ reads
           ▼
┌──────────────────────────────┐
│  config.py                   │
│  llm_model = "llama-3.3-70b-versatile"  │
│  embedding_model             │
│  chunk_size / overlap        │
│  retrieval_k                 │
│  groq_api_key (from .env)   │
└──────────────────────────────┘
```

---

##  Configuration

All settings live in `config.py`:

| Parameter | Default | Description |
|:---|:---|:---|
| `llm_model` | `llama-3.3-70b-versatile` | Groq model — change here only |
| `embedding_model` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `chunk_size` | 800 | Characters per text chunk |
| `chunk_overlap` | 150 | Overlap between adjacent chunks |
| `retrieval_k` | 4 | Chunks retrieved per query |
| `llm_temperature` | 0.2 | Response creativity (lower = more factual) |

---

## 🛠️ Tech Stack

| Layer | Library |
|:---|:---|
| UI | Streamlit ≥ 1.35 |
| LLM | Groq API · `langchain-groq` |
| Model | `llama-3.3-70b-versatile` |
| Embeddings | `all-MiniLM-L6-v2` · `langchain-huggingface` |
| Vector store | FAISS (CPU) · `langchain-community` |
| Text splitting | `langchain-text-splitters` |
| PDF parsing | PyPDF · `langchain-community` |
| Env management | python-dotenv |

---

## 🔮 Future Improvements

- [ ] Multi-PDF support
- [ ] Streaming responses with `st.write_stream`
- [ ] Persistent FAISS index (save/load to disk)
- [ ] Conversation memory for follow-up questions
- [ ] OCR support for scanned PDFs (pytesseract)
- [ ] Re-ranking for improved retrieval (BGE, Cohere)
- [ ] Deploy to Streamlit Community Cloud

---

## 📄 License

MIT
