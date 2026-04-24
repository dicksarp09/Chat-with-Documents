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

    embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    embedding_dimension: int = 384  # MiniLM outputs 384 dimensions
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Use smaller models for memory-constrained environments
    embedding_model_tiny: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Auto-select smaller model for cloud/free tier
    embedding_model_cloud: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Lite mode: disable embeddings (BM25 only) for free tier
    use_lite_mode: bool = os.getenv("LITE_MODE", "").lower() == "true"

    # Disable cross-encoder reranker to save memory (~200MB)
    use_reranker: bool = os.getenv("USE_RERANKER", "false").lower() == "true"

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
    rerank_top_k: int = 5
    hybrid_alpha: float = 0.5  # Balanced - will use both dense and sparse

    # Precision tuning
    min_retrieval_score: float = 0.05  # Lower for dense (scores around 0.1-0.2)
    min_rerank_score: float = 0.3  # Higher threshold for quality

    # Diversity settings
    enable_diversity_filter: bool = True
    fallback_answer: str = (
        "I couldn't find relevant information to answer your question."
    )

    log_level: str = "INFO"

    # Performance settings
    cache_enabled: bool = True
    cache_max_size: int = 100
    cache_ttl_seconds: int = 3600

    # Pre-load models at startup
    warmup_on_init: bool = True
    warmup_batch_size: int = 32

    # Error handling
    enable_fallback: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Metrics collection
    collect_metrics: bool = True
    metrics_max_history: int = 1000

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
