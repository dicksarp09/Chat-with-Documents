import sys
sys.path.insert(0, ".")

import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

@dataclass
class StageMetrics:
    stage: str
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    query: str
    intent: str
    latencies: List[StageMetrics]
    precision: float
    faithfulness: float
    relevance: float
    groundedness: float
    answer: str
    sources_count: int
    total_latency_ms: float

class ImprovedPipelineEvaluator:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc_id = None
        self.setup_pipeline()

    def setup_pipeline(self):
        """Initialize pipeline components and process PDF"""
        from parsers.pdf_parser import parse_pdf
        from chunking.hierarchical_chunker import chunk_document
        from embeddings.embedder import get_embedder
        from storage.vector_store import get_vector_store
        from storage.sqlite_store import get_sqlite_store
        from retrieval.hybrid_retriever import get_hybrid_retriever
        from retrieval.query_classifier import get_intent_classifier

        logger.info("=" * 60)
        logger.info("IMPROVED PDF PIPELINE EVALUATION")
        logger.info("=" * 60)

        # Parse PDF
        t0 = time.time()
        self.parsed_doc = parse_pdf(self.pdf_path)
        parse_time = (time.time() - t0) * 1000
        logger.info(f"[1] PDF Parsing: {parse_time:.0f}ms ({len(self.parsed_doc.sections)} sections)")
        self.parse_latency = parse_time

        # Chunk
        t0 = time.time()
        self.structure = chunk_document(self.parsed_doc)
        chunk_time = (time.time() - t0) * 1000
        logger.info(f"[2] Chunking: {chunk_time:.0f}ms ({self.structure.total_chunks} leaf chunks)")
        self.chunk_latency = chunk_time
        self.doc_id = self.parsed_doc.doc_id

        # Embed
        t0 = time.time()
        self.embedder = get_embedder()
        self.texts = [node.text for node in self.structure.nodes]
        self.embeddings = self.embedder.encode(self.texts, show_progress=True, batch_size=64)
        embed_time = (time.time() - t0) * 1000
        logger.info(f"[3] Embedding: {embed_time:.0f}ms ({len(self.embeddings)} vectors, batch=64)")
        self.embed_latency = embed_time

        # Vector store
        t0 = time.time()
        self.vector_store = get_vector_store()
        self.vector_store.add_nodes(self.structure.nodes)
        vs_time = (time.time() - t0) * 1000
        logger.info(f"[4] Vector Store: {vs_time:.0f}ms")
        self.vs_latency = vs_time

        # Hybrid retriever
        t0 = time.time()
        self.retriever = get_hybrid_retriever()
        self.retriever.rebuild_index(self.doc_id)
        retrieval_setup_time = (time.time() - t0) * 1000
        logger.info(f"[5] Retriever Setup: {retrieval_setup_time:.0f}ms")

        # Query classifier
        classifier = get_intent_classifier()
        logger.info(f"[6] Query Classifier: Ready")

        # SQLite
        t0 = time.time()
        self.sqlite = get_sqlite_store()
        chunks_data = [
            {"id": node.id, "doc_id": node.doc_id, "text": node.text,
             "section_title": node.section_title, "level": node.level, "parent_id": node.parent_id}
            for node in self.structure.nodes
        ]
        self.sqlite.save_dataset(self.doc_id, self.pdf_path, "document", chunks_data)
        chunk_emb = {node.id: emb for node, emb in zip(self.structure.nodes, self.embeddings)}
        self.sqlite.save_embeddings(chunk_emb)
        sqlite_time = (time.time() - t0) * 1000
        logger.info(f"[7] SQLite Save: {sqlite_time:.0f}ms")

        self.setup_latency = parse_time + chunk_time + embed_time + vs_time + retrieval_setup_time + sqlite_time
        logger.info(f"Total Setup: {self.setup_latency:.0f}ms")

    def evaluate_query(self, query: str) -> EvaluationResult:
        """Evaluate a single query through the pipeline"""
        from retrieval.reranker import get_reranker
        from compression.compressor import get_compressor
        from llm.reasoning import get_reasoning_pipeline
        from retrieval.query_classifier import get_intent_classifier

        classifier = get_intent_classifier()
        intent = classifier.classify(query)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"QUERY: {query}")
        logger.info(f"INTENT: {intent.value}")
        logger.info(f"{'=' * 60}")

        latencies = []
        total_t0 = time.time()

        # Retrieval
        t0 = time.time()
        results = self.retriever.retrieve(query, doc_id=self.doc_id, top_k=20)
        retrieval_time = (time.time() - t0) * 1000
        latencies.append(StageMetrics("retrieval", retrieval_time, {"results": len(results)}))
        logger.info(f"  Retrieval: {retrieval_time:.0f}ms ({len(results)} results)")

        # Reranking
        t0 = time.time()
        reranker = get_reranker()
        reranked = reranker.rerank(query, results, top_k=5)
        rerank_time = (time.time() - t0) * 1000
        latencies.append(StageMetrics("reranking", rerank_time, {"results": len(reranked)}))
        logger.info(f"  Reranking: {rerank_time:.0f}ms ({len(reranked)} results)")

        # Compression
        t0 = time.time()
        compressor = get_compressor()
        context = compressor.compress(query, reranked)
        compression_time = (time.time() - t0) * 1000
        latencies.append(StageMetrics("compression", compression_time, {
            "original": context.original_length,
            "compressed": context.compressed_length,
            "ratio": context.compression_ratio
        }))
        logger.info(f"  Compression: {compression_time:.0f}ms (ratio: {context.compression_ratio:.1%})")

        # Generation
        t0 = time.time()
        reasoning = get_reasoning_pipeline()
        result = reasoning.query_analysis(query, context)
        generation_time = (time.time() - t0) * 1000
        latencies.append(StageMetrics("generation", generation_time, {"answer_len": len(result.answer)}))
        logger.info(f"  Generation: {generation_time:.0f}ms")

        total_latency = (time.time() - total_t0) * 1000
        latencies.append(StageMetrics("total", total_latency, {}))

        # Calculate metrics
        precision = self._calculate_precision(results)
        faithfulness = self._calculate_faithfulness(result.answer, context.text, results)
        relevance = self._calculate_relevance(query, result.answer)
        groundedness = self._calculate_groundedness(result.answer, context.text)

        logger.info(f"\n  Precision: {precision:.3f}")
        logger.info(f"  Faithfulness: {faithfulness:.3f}")
        logger.info(f"  Relevance: {relevance:.3f}")
        logger.info(f"  Groundedness: {groundedness:.3f}")
        logger.info(f"  Total Latency: {total_latency:.0f}ms")

        return EvaluationResult(
            query=query,
            intent=intent.value,
            latencies=latencies,
            precision=precision,
            faithfulness=faithfulness,
            relevance=relevance,
            groundedness=groundedness,
            answer=result.answer,
            sources_count=len(result.sources),
            total_latency_ms=total_latency
        )

    def _calculate_precision(self, results: List[Dict]) -> float:
        """Context Precision: High scores for top results indicate precision"""
        if not results:
            return 0.0
        scores = [r.get("score", 0) for r in results[:10]]
        return min(sum(scores) / len(scores), 1.0) if scores else 0.0

    def _calculate_faithfulness(self, answer: str, context: str, results: List[Dict]) -> float:
        """Faithfulness: Answer is supported by retrieved context"""
        if not answer or not context:
            return 0.0
        answer_words = set(w.lower() for w in answer.split() if len(w) > 4)
        context_words = set(context.lower().split())
        if not answer_words:
            return 1.0
        matches = sum(1 for w in answer_words if w in context_words)
        return matches / len(answer_words) if answer_words else 0.0

    def _calculate_relevance(self, query: str, answer: str) -> float:
        """Relevance: Answer addresses the query"""
        if not query or not answer:
            return 0.0
        query_terms = set(w.lower() for w in query.split() if len(w) > 3)
        answer_lower = answer.lower()
        matches = sum(1 for term in query_terms if term in answer_lower)
        return matches / len(query_terms) if query_terms else 0.0

    def _calculate_groundedness(self, answer: str, context: str) -> float:
        """Groundedness: Answer claims can be verified in context"""
        if not answer or not context:
            return 0.5
        answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]
        if not answer_sentences:
            return 1.0
        supported = 0
        for sent in answer_sentences:
            words = set(w.lower() for w in sent.split() if len(w) > 3)
            if words and any(w in context.lower() for w in words):
                supported += 1
        return supported / len(answer_sentences) if answer_sentences else 0.5

def main():
    import os

    pdf_path = "22705iied.pdf"
    if not os.path.exists(pdf_path):
        pdf_path = "DICKSON SARPONG_AI Engineer.pdf"

    if not os.path.exists(pdf_path):
        logger.error("No PDF found!")
        return

    logger.info(f"Evaluating: {pdf_path}")

    evaluator = ImprovedPipelineEvaluator(pdf_path)

    queries = [
        ("Who is at risk from food security issues?", "EXTRACTION"),
        ("What are the main causes of food insecurity?", "REASONING"),
        ("What recommendations does the document make?", "RECOMMENDATION"),
        ("How does climate change affect food production?", "REASONING"),
        ("What regions are most affected by food insecurity?", "EXTRACTION"),
    ]

    results = []
    for q, expected_intent in queries:
        try:
            result = evaluator.evaluate_query(q)
            results.append({
                "query": q,
                "intent": result.intent,
                "expected_intent": expected_intent,
                "intent_match": result.intent == expected_intent,
                "precision": result.precision,
                "faithfulness": result.faithfulness,
                "relevance": result.relevance,
                "groundedness": result.groundedness,
                "latency_ms": result.total_latency_ms,
                "answer_preview": result.answer[:300]
            })
        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            import traceback
            traceback.print_exc()

    if results:
        avg_precision = sum(r["precision"] for r in results) / len(results)
        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
        avg_relevance = sum(r["relevance"] for r in results) / len(results)
        avg_groundedness = sum(r["groundedness"] for r in results) / len(results)
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        total_latency = sum(r["latency_ms"] for r in results)
        intent_accuracy = sum(1 for r in results if r["intent_match"]) / len(results)

        logger.info(f"\n{'=' * 60}")
        logger.info("IMPROVED PIPELINE - EVALUATION RESULTS")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Precision:     {avg_precision:.3f}")
        logger.info(f"  Faithfulness:  {avg_faithfulness:.3f}")
        logger.info(f"  Relevance:     {avg_relevance:.3f}")
        logger.info(f"  Groundedness:  {avg_groundedness:.3f}")
        logger.info(f"  Avg Latency:   {avg_latency:.0f}ms")
        logger.info(f"  Intent Match:  {intent_accuracy:.0%}")

        overall = (avg_precision + avg_faithfulness + avg_relevance + avg_groundedness) / 4
        if overall >= 0.9: grade = "A - EXCELLENT"
        elif overall >= 0.8: grade = "B - GOOD"
        elif overall >= 0.7: grade = "C - SATISFACTORY"
        elif overall >= 0.6: grade = "D - NEEDS IMPROVEMENT"
        else: grade = "F - POOR"
        logger.info(f"  OVERALL:       {overall:.3f} ({grade})")

        output = {
            "timestamp": datetime.now().isoformat(),
            "document": pdf_path,
            "doc_id": evaluator.doc_id,
            "setup_latency_ms": evaluator.setup_latency,
            "aggregate": {
                "precision": avg_precision,
                "faithfulness": avg_faithfulness,
                "relevance": avg_relevance,
                "groundedness": avg_groundedness,
                "avg_latency_ms": avg_latency,
                "total_latency_ms": total_latency,
                "intent_accuracy": intent_accuracy
            },
            "improvements": {
                "query_intent_classification": "enabled",
                "query_expansion": "enabled",
                "citation_enforced_prompts": "enabled",
                "embedding_cache": "enabled",
                "optimized_batch_size": 64
            },
            "per_query": results
        }

        output_file = f"improved_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()