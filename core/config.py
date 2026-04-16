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

    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 200
    max_chunk_size: int = 400

    retrieval_top_k: int = 15
    rerank_top_k: int = 5
    hybrid_alpha: float = 0.6

    # Precision tuning (balanced approach)
    min_retrieval_score: float = 0.05
    min_rerank_score: float = -5.0
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
