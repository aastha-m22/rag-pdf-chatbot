"""
config.py
---------
Single source of truth for ALL project configuration.

Rules enforced here:
  - Model name defined ONCE — never referenced as a string anywhere else
  - API key loaded from environment only — never hardcoded
  - Every tunable has a clear comment

Import pattern everywhere else:
    from config import config
    model = config.llm_model   ← always via this object
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env before any field default_factory runs
load_dotenv()


@dataclass
class Config:
    # ── LLM (Groq) ─────────────────────────────────────────────────────────
    # ONLY place the model string is defined — all other files use config.llm_model
    llm_model: str = "llama-3.3-70b-versatile"

    llm_temperature: float = 0.2      # Lower = more factual, less creative
    llm_max_tokens: int = 1024        # Max tokens in the generated answer

    # ── Embeddings ──────────────────────────────────────────────────────────
    # sentence-transformers model; runs locally on CPU, no API key needed
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Chunking ────────────────────────────────────────────────────────────
    chunk_size: int = 800             # Characters per chunk
    chunk_overlap: int = 150          # Overlap keeps context across chunk boundaries

    # ── Retrieval ───────────────────────────────────────────────────────────
    retrieval_k: int = 4              # How many chunks to retrieve per query

    # ── API Key ─────────────────────────────────────────────────────────────
    # Read from GROQ_API_KEY environment variable / .env file
    # The UI can override this at runtime (sidebar input)
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )

    # ── System prompt ────────────────────────────────────────────────────────
    system_prompt: str = (
        "You are a precise, helpful assistant. Answer the user's question "
        "using ONLY the information provided in the context below. "
        "If the answer is not present in the context, clearly say so — "
        "do not invent or guess.\n\n"
        "Context:\n{context}"
    )

    def validate(self) -> None:
        """Raise ValueError with a clear message if critical config is missing."""
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "  → Add it to your .env file:  GROQ_API_KEY=gsk_...\n"
                "  → Or paste it in the sidebar API Key field."
            )


# Module-level singleton — the one object every module imports
config = Config()
