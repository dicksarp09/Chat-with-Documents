import pytest
import sys
sys.path.insert(0, ".")

import numpy as np
from core.cache import QueryCache, MetricsCollector


class TestQueryCache:
    def test_cache_miss_returns_none(self):
        cache = QueryCache(max_size=10, ttl_seconds=60)
        result = cache.get("test query", "doc123")
        assert result is None

    def test_cache_set_and_get(self):
        cache = QueryCache(max_size=10, ttl_seconds=60)
        cache.set("test query", "doc123", {"answer": "test answer"})

        result = cache.get("test query", "doc123")
        assert result is not None
        assert result["answer"] == "test answer"

    def test_cache_different_doc_ids(self):
        cache = QueryCache(max_size=10, ttl_seconds=60)

        cache.set("same query", "doc1", {"answer": "answer1"})
        cache.set("same query", "doc2", {"answer": "answer2"})

        result1 = cache.get("same query", "doc1")
        result2 = cache.get("same query", "doc2")

        assert result1["answer"] == "answer1"
        assert result2["answer"] == "answer2"

    def test_cache_eviction(self):
        cache = QueryCache(max_size=2, ttl_seconds=60)

        cache.set("query1", "doc1", {"answer": "a1"})
        cache.set("query2", "doc2", {"answer": "a2"})
        cache.set("query3", "doc3", {"answer": "a3"})

        assert cache.get("query1", "doc1") is None

    def test_cache_stats(self):
        cache = QueryCache(max_size=10, ttl_seconds=60)

        cache.set("q1", "d1", {"a": "1"})
        cache.get("q2", "d2")
        cache.get("q1", "d1")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestMetricsCollector:
    def test_record_and_get_stats(self):
        metrics = MetricsCollector()

        metrics.record("query1", "doc1", 1000.0, True)
        metrics.record("query2", "doc2", 2000.0, True)
        metrics.record("query3", "doc3", 3000.0, False, "error")

        stats = metrics.get_stats()
        assert stats["total_queries"] == 3
        assert stats["success_rate"] == pytest.approx(2/3)
        assert stats["avg_latency_ms"] == 2000.0

    def test_percentiles(self):
        metrics = MetricsCollector()

        for i in range(100):
            metrics.record(f"q{i}", "d{i}", float(i * 100), True)

        stats = metrics.get_stats()
        assert stats["p50_latency_ms"] == pytest.approx(5000.0, rel=500)
        assert stats["p95_latency_ms"] == pytest.approx(9500.0, rel=500)


class TestEmbedder:
    def test_embedder_returns_numpy(self):
        from embeddings.embedder import get_embedder

        embedder = get_embedder()
        result = embedder.encode("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape[-1] == 384  # Check last dim (embedding dim)

    def test_embedder_batch(self):
        from embeddings.embedder import get_embedder

        embedder = get_embedder()
        result = embedder.encode(["text1", "text2", "text3"])

        assert result.shape == (3, 384)

    def test_similarity(self):
        from embeddings.embedder import get_embedder

        embedder = get_embedder()
        emb1 = embedder.encode("hello world")
        emb2 = embedder.encode("hello world")

        sim = embedder.similarity(emb1, emb2)
        assert sim > 0.9


class TestVectorStore:
    def test_vector_store_add_and_query(self):
        from storage.vector_store import get_vector_store
        from chunking.hierarchical_chunker import ChunkNode
        import uuid

        store = get_vector_store()

        node = ChunkNode(
            id=str(uuid.uuid4()),
            doc_id="test-doc",
            text="This is a test document about machine learning.",
            section_title="Test Section",
            level=1,
        )

        store.add_nodes([node])
        results = store.query_dense("machine learning", top_k=1)

        assert len(results) >= 1


class TestHybridRetriever:
    def test_retriever_initializes(self):
        from retrieval.hybrid_retriever import get_hybrid_retriever

        retriever = get_hybrid_retriever()
        assert retriever is not None


class TestQueryClassifier:
    def test_classify_extraction(self):
        from retrieval.query_classifier import classify_query_intent

        intent = classify_query_intent("What is the capital of France?")
        assert intent == "extraction"

    def test_classify_reasoning(self):
        from retrieval.query_classifier import classify_query_intent

        intent = classify_query_intent("Why does climate change affect crops?")
        assert intent == "reasoning"

    def test_classify_recommendation(self):
        from retrieval.query_classifier import classify_query_intent

        intent = classify_query_intent("What should we do about food security?")
        assert intent == "recommendation"


class TestRAGASEvaluator:
    def test_context_precision(self):
        from evaluation.ragas import calculate_context_precision

        chunks = [
            {"text": "climate change affects agriculture"},
            {"text": "food security is important"},
            {"text": "machine learning is growing"},
        ]

        precision = calculate_context_precision("climate food", chunks)
        assert precision > 0

    def test_faithfulness(self):
        from evaluation.ragas import calculate_faithfulness

        answer = "Climate change affects crop yields."
        context = "Climate change affects crop yields in vulnerable regions."

        faith = calculate_faithfulness(answer, context)
        assert faith > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])