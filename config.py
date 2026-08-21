# config.py — Centralized configuration for RAG PDF Chatbot

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K_RESULTS = 4
SIMILARITY_THRESHOLD = 0.7

# --- LLM ---
LLM_MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.0
MAX_TOKENS = 512

# --- Vector Store ---
VECTORSTORE_PATH = "vectorstore/"
PERSIST = True
