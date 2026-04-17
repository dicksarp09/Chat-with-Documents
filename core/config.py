import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    app_name: str = "Document Intelligence Engine"
    version: str = "1.0.0"

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = "llama-3.3-70b-versatile"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Use smaller models for memory-constrained environments
    embedding_model_tiny: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Cloud detection
    is_cloud: bool = (
        os.getenv("RENDER", "").lower() == "true"
        or os.getenv("VERCEL", "").lower() == "true"
    )

    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 200
    max_chunk_size: int = 400

    retrieval_top_k: int = 20
    rerank_top_k: int = 15
    hybrid_alpha: float = 0.65

    # Precision tuning (balanced for better precision while maintaining recall)
    min_retrieval_score: float = 0.10
    min_rerank_score: float = -10.0  # Disabled - let reranker decide

    # Diversity settings (set to False for precision over recall)
    enable_diversity_filter: bool = False
    fallback_answer: str = (
        "I couldn't find relevant information to answer your question."
    )

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
