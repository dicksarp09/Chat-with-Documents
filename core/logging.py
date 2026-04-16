import logging
import json
import time
import uuid
from typing import Any, Dict, Optional
from datetime import datetime
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

logger = logging.getLogger(__name__)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "trace_id": trace_id_var.get(),
        }

        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = round(record.latency_ms, 2)

        if hasattr(record, "component"):
            log_data["component"] = record.component

        if hasattr(record, "success"):
            log_data["success"] = record.success

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TrackedComponent:
    """Context manager for tracking component latency"""

    def __init__(self, component_name: str, log_level: int = logging.INFO):
        self.component_name = component_name
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.perf_counter() - self.start_time) * 1000
        success = exc_type is None

        extra = {
            "component": self.component_name,
            "latency_ms": latency_ms,
            "success": success,
        }

        if exc_val:
            extra["error"] = str(exc_val)

        logger.log(
            self.log_level,
            f"{self.component_name} completed in {latency_ms:.2f}ms",
            extra=extra,
        )

        return False


class RequestTracker:
    """Tracks request lifecycle"""

    def __init__(self, request_id: str = None, query: str = ""):
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.query = query[:100] if query else ""
        self.start_time = time.perf_counter()
        self.token = request_id_var.set(self.request_id)
        self.trace_token = trace_id_var.set(str(uuid.uuid4()))

    def __enter__(self):
        logger.info(f"Request {self.request_id} started", extra={"query": self.query})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        total_ms = (time.perf_counter() - self.start_time) * 1000
        success = exc_type is None

        request_id_var.reset(self.token)
        trace_id_var.reset(self.trace_token)

        logger.info(
            f"Request {self.request_id} {'failed' if not success else 'completed'} in {total_ms:.2f}ms",
            extra={
                "request_id": self.request_id,
                "latency_ms": total_ms,
                "success": success,
                "error": str(exc_val) if exc_val else None,
            },
        )

        return False


def setup_structured_logging(level: str = "INFO"):
    """Setup structured logging for the application"""
    formatter = StructuredFormatter()

    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


def log_retrieval_results(results: list, query: str, stage: str = "hybrid"):
    """Log retrieval results with stats"""
    scores = [r.get("score", 0) for r in results]
    logger.info(
        f"Retrieval ({stage}): {len(results)} results",
        extra={
            "component": "retrieval",
            "query": query[:100],
            "result_count": len(results),
            "top_score": max(scores) if scores else 0,
            "avg_score": sum(scores) / len(scores) if scores else 0,
        },
    )


def log_llm_call(
    model: str, prompt_length: int, response_length: int, latency_ms: float
):
    """Log LLM call metrics"""
    logger.info(
        f"LLM call completed",
        extra={
            "component": "llm",
            "model": model,
            "prompt_tokens": prompt_length,
            "completion_tokens": response_length,
            "latency_ms": latency_ms,
        },
    )


def log_pipeline_stage(stage: str, latency_ms: float, item_count: int = None):
    """Log pipeline stage latency"""
    logger.info(
        f"Pipeline stage: {stage}",
        extra={
            "component": stage,
            "latency_ms": latency_ms,
            "item_count": item_count,
        },
    )
