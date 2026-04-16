import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import threading


@dataclass
class PipelineMetrics:
    request_id: str = ""
    query: str = ""
    start_time: float = field(default_factory=time.perf_counter)

    retrieval_latency_ms: float = 0
    rerank_latency_ms: float = 0
    compression_latency_ms: float = 0
    generation_latency_ms: float = 0
    total_latency_ms: float = 0

    retrieval_results_count: int = 0
    reranked_results_count: int = 0
    compression_ratio: float = 0

    success: bool = True
    error: str = ""

    def finish(self):
        self.total_latency_ms = (time.perf_counter() - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query[:100] if self.query else "",
            "latency_ms": {
                "retrieval": round(self.retrieval_latency_ms, 2),
                "reranking": round(self.rerank_latency_ms, 2),
                "compression": round(self.compression_latency_ms, 2),
                "generation": round(self.generation_latency_ms, 2),
                "total": round(self.total_latency_ms, 2),
            },
            "counts": {
                "retrieved": self.retrieval_results_count,
                "reranked": self.reranked_results_count,
                "compression_ratio": round(self.compression_ratio, 2),
            },
            "success": self.success,
            "error": self.error,
        }


class MetricsCollector:
    """Collects and aggregates metrics"""

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: list[PipelineMetrics] = []
        self._aggregates: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "count": 0,
                "total_latency": 0,
                "success_count": 0,
                "failure_count": 0,
            }
        )

    def add(self, metrics: PipelineMetrics):
        with self._lock:
            self._metrics.append(metrics)
            self._update_aggregates(metrics)

    def _update_aggregates(self, m: PipelineMetrics):
        agg = self._aggregates["overall"]
        agg["count"] += 1
        agg["total_latency"] += m.total_latency_ms
        if m.success:
            agg["success_count"] += 1
        else:
            agg["failure_count"] += 1

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._aggregates["overall"]["count"]:
                return {"status": "no_data"}

            agg = self._aggregates["overall"]
            count = agg["count"]

            return {
                "total_requests": count,
                "success_rate": round(agg["success_count"] / count * 100, 1),
                "avg_latency_ms": round(agg["total_latency"] / count, 1),
                "failure_count": agg["failure_count"],
            }

    def get_recent(self, n: int = 10) -> list[Dict[str, Any]]:
        with self._lock:
            recent = self._metrics[-n:]
            return [m.to_dict() for m in recent]

    def clear(self):
        with self._lock:
            self._metrics.clear()
            self._aggregates.clear()


_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


class Timer:
    """Simple timer context manager for latency tracking"""

    def __init__(self):
        self.start = 0
        self.elapsed_ms = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


class RetrievalLatencyTracker:
    """Tracks retrieval-specific metrics"""

    def __init__(self):
        self.dense_latency = 0
        self.sparse_latency = 0
        self.merge_latency = 0

    def record_dense(self, ms: float):
        self.dense_latency = ms

    def record_sparse(self, ms: float):
        self.sparse_latency = ms

    def record_merge(self, ms: float):
        self.merge_latency = ms

    def total(self) -> float:
        return self.dense_latency + self.sparse_latency + self.merge_latency


class QualityTracker:
    """Tracks retrieval quality metrics"""

    def __init__(self):
        self.scores: list[float] = []

    def record(self, score: float):
        self.scores.append(score)

    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0

    def top_score(self) -> float:
        return max(self.scores) if self.scores else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_score": round(self.avg_score(), 3),
            "top_score": round(self.top_score(), 3),
            "count": len(self.scores),
        }
