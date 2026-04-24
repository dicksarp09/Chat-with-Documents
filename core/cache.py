import logging
import time
from typing import Any, Dict, Optional, Callable
from functools import lru_cache
import hashlib
import json
import threading

logger = logging.getLogger(__name__)


class QueryCache:
    """In-memory query result cache with TTL support."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, doc_id: str) -> str:
        content = f"{doc_id}:{query}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, doc_id: str) -> Optional[Dict[str, Any]]:
        key = self._make_key(query, doc_id)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["timestamp"] < self._ttl_seconds:
                    self._hits += 1
                    logger.debug(f"Cache hit: {query[:30]}...")
                    return entry["result"]
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    def set(self, query: str, doc_id: str, result: Dict[str, Any]) -> None:
        key = self._make_key(query, doc_id)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k]["timestamp"]
                )
                del self._cache[oldest_key]

            self._cache[key] = {
                "result": result,
                "timestamp": time.time(),
            }
            logger.debug(f"Cache set: {query[:30]}...")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


class EmbeddingCache:
    """In-memory cache for embeddings."""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, text: str) -> Optional[Any]:
        key = hashlib.md5(text.encode()).hexdigest()
        with self._lock:
            return self._cache.get(key)

    def set(self, text: str, embedding: Any) -> None:
        key = hashlib.md5(text.encode()).hexdigest()
        with self._lock:
            if len(self._cache) >= self._max_size:
                keys = list(self._cache.keys())
                del self._cache[keys[0]]
            self._cache[key] = embedding


class RetrievalCache:
    """In-memory cache for retrieval results."""

    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, query: str, doc_id: str) -> Optional[Any]:
        key = hashlib.md5(f"{doc_id}:{query}".encode()).hexdigest()
        with self._lock:
            return self._cache.get(key)

    def set(self, query: str, doc_id: str, results: Any) -> None:
        key = hashlib.md5(f"{doc_id}:{query}".encode()).hexdigest()
        with self._lock:
            if len(self._cache) >= self._max_size:
                keys = list(self._cache.keys())
                del self._cache[keys[0]]
            self._cache[key] = results


class MetricsCollector:
    """Collects query latency and success rate metrics."""

    def __init__(self):
        self._queries: list = []
        self._lock = threading.Lock()
        self._max_history = 1000

    def record(
        self,
        query: str,
        doc_id: str,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._queries.append({
                "query": query[:100],
                "doc_id": doc_id,
                "latency_ms": latency_ms,
                "success": success,
                "error": error,
                "timestamp": time.time(),
            })

            if len(self._queries) > self._max_history:
                self._queries = self._queries[-self._max_history:]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self._queries:
                return {
                    "total_queries": 0,
                    "success_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                }

            successful = [q for q in self._queries if q["success"]]
            latencies = sorted([q["latency_ms"] for q in self._queries])

            n = len(latencies)
            return {
                "total_queries": len(self._queries),
                "success_rate": len(successful) / len(self._queries),
                "avg_latency_ms": sum(latencies) / n,
                "p50_latency_ms": latencies[int(n * 0.5)],
                "p95_latency_ms": latencies[int(n * 0.95)],
                "p99_latency_ms": latencies[int(n * 0.99)],
            }

    def get_recent(self, limit: int = 10) -> list:
        with self._lock:
            return self._queries[-limit:]


_query_cache: Optional[QueryCache] = None
_embedding_cache: Optional[EmbeddingCache] = None
_retrieval_cache: Optional[RetrievalCache] = None
_metrics: Optional[MetricsCollector] = None


def get_query_cache() -> QueryCache:
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache()
    return _query_cache


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


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics