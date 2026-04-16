"""
Redis-based cache for embeddings and retrieval results.
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - caching disabled")


class EmbeddingCache:
    """Cache for query embeddings."""

    def __init__(self, host: str = None, port: int = None, ttl: int = 3600):
        self.enabled = False
        self.ttl = ttl
        self.redis = None

        if REDIS_AVAILABLE:
            try:
                host = host or os.getenv("REDIS_HOST", "localhost")
                port = port or int(os.getenv("REDIS_PORT", "6379"))
                self.redis = redis.Redis(host=host, port=port, decode_responses=False)
                self.redis.ping()
                self.enabled = True
                logger.info(f"Embedding cache enabled (Redis at {host}:{port})")
            except Exception as e:
                logger.warning(f"Redis not available: {e}. Caching disabled.")
        else:
            logger.warning("redis-py not installed. Caching disabled.")

    def _hash_key(self, text: str) -> str:
        return f"embed:{hashlib.md5(text.encode()).hexdigest()}"

    def get(self, text: str) -> Optional[np.ndarray]:
        if not self.enabled or not self.redis:
            return None

        try:
            key = self._hash_key(text)
            data = self.redis.get(key)
            if data:
                logger.info(f"Cache hit for embedding: {text[:30]}...")
                return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None

    def set(self, text: str, embedding: np.ndarray):
        if not self.enabled or not self.redis:
            return

        try:
            key = self._hash_key(text)
            data = embedding.astype(np.float32).tobytes()
            self.redis.setex(key, self.ttl, data)
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")


class RetrievalCache:
    """Cache for retrieval results."""

    def __init__(self, host: str = None, port: int = None, ttl: int = 1800):
        self.enabled = False
        self.ttl = ttl
        self.redis = None

        if REDIS_AVAILABLE:
            try:
                host = host or os.getenv("REDIS_HOST", "localhost")
                port = port or int(os.getenv("REDIS_PORT", "6379"))
                self.redis = redis.Redis(host=host, port=port, decode_responses=True)
                self.redis.ping()
                self.enabled = True
                logger.info(f"Retrieval cache enabled (Redis at {host}:{port})")
            except Exception as e:
                logger.warning(f"Redis not available: {e}. Caching disabled.")
        else:
            logger.warning("redis-py not installed. Caching disabled.")

    def _hash_key(self, query: str, dataset_id: str, top_k: int) -> str:
        return (
            f"retrieval:{dataset_id}:{top_k}:{hashlib.md5(query.encode()).hexdigest()}"
        )

    def get(
        self, query: str, dataset_id: str, top_k: int = 20
    ) -> Optional[List[Dict[str, Any]]]:
        if not self.enabled or not self.redis:
            return None

        try:
            key = self._hash_key(query, dataset_id, top_k)
            data = self.redis.get(key)
            if data:
                logger.info(f"Cache hit for retrieval: {query[:30]}...")
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None

    def set(
        self, query: str, dataset_id: str, top_k: int, results: List[Dict[str, Any]]
    ):
        if not self.enabled or not self.redis:
            return

        try:
            key = self._hash_key(query, dataset_id, top_k)
            # Store simplified results (just IDs and scores)
            simplified = [
                {"node_id": r.get("node_id", ""), "score": r.get("score", 0)}
                for r in results
            ]
            self.redis.setex(key, self.ttl, json.dumps(simplified))
            logger.info(f"Cached retrieval: {len(results)} results")
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    def clear_dataset(self, dataset_id: str):
        """Clear cache for a specific dataset."""
        if not self.enabled or not self.redis:
            return

        try:
            pattern = f"retrieval:{dataset_id}:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries for {dataset_id}")
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")

    def clear_all(self):
        """Clear all caches."""
        if not self.enabled or not self.redis:
            return

        try:
            patterns = ["embed:*", "retrieval:*"]
            for pattern in patterns:
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
            logger.info("Cleared all caches")
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")


def get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_retrieval_cache() -> RetrievalCache:
    global _retrieval_cache
    if _retrieval_cache is None:
        _retrieval_cache = RetrievalCache()
    return _retrieval_cache


_embedding_cache: Optional[EmbeddingCache] = None
_retrieval_cache: Optional[RetrievalCache] = None
