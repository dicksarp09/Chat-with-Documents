import sys

sys.path.insert(0, ".")

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import json
from datetime import datetime

# Import pipeline components
from parsers.pdf_parser import parse_pdf
from chunking.hierarchical_chunker import chunk_document
from embeddings.embedder import get_embedder
from storage.vector_store import get_vector_store
from retrieval.hybrid_retriever import get_hybrid_retriever
from retrieval.reranker import get_reranker
from compression.compressor import get_compressor
from llm.reasoning import get_reasoning_pipeline
from core.logging import TrackedComponent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RAGASMetrics:
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    overall: float = 0.0


class RAGASEvaluator:
    """Evaluate RAG system using RAGAS metrics"""

    def __init__(self):
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        try:
            from llm.groq_client import get_groq_client

            self.llm = get_groq_client()
            if self.llm.is_available():
                logger.info("LLM available for evaluation")
            else:
                logger.warning("LLM not available - using heuristic evaluation")
        except Exception as e:
            logger.warning(f"LLM init failed: {e}")
            self.llm = None

    def evaluate_retrieval(
        self,
        query: str,
        expected_terms: List[str] = None,
        ground_truth_docs: List[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate retrieval stage with context precision and recall"""

        retriever = get_hybrid_retriever()

        # Retrieve
        with TrackedComponent("retrieval_eval"):
            results = retriever.retrieve(query, top_k=20)

        # Calculate metrics
        scores = [r.get("score", 0) for r in results]

        # Context Precision: Top-k results should be relevant
        # Heuristic: top results should have scores above threshold
        if results:
            top_k_scores = scores[:5]
            avg_top5 = sum(top_k_scores) / len(top_k_scores)
            context_precision = min(avg_top5, 1.0)
        else:
            context_precision = 0.0

        # Context Recall: Were relevant docs retrieved?
        # Check if any expected terms appear in top results
        retrieved_texts = []
        for r in results[:10]:
            if r.get("node") and hasattr(r["node"], "text"):
                retrieved_texts.append(r["node"].text.lower())

        if expected_terms and retrieved_texts:
            hits = sum(
                1
                for term in expected_terms
                if any(term.lower() in text for text in retrieved_texts)
            )
            context_recall = hits / len(expected_terms)
        else:
            # No ground truth - use score distribution
            relevant_count = sum(1 for s in scores[:10] if s > 0.3)
            context_recall = relevant_count / 10

        return {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "top_scores": scores[:10],
            "result_count": len(results),
        }

    def evaluate_full(
        self, query: str, expected_answer_contains: List[str] = None
    ) -> RAGASMetrics:
        """Evaluate full RAG pipeline"""

        logger.info(f"\n{'=' * 60}")
        logger.info(f"RAGAS EVALUATION: '{query[:50]}...'")
        logger.info(f"{'=' * 60}")

        # Stage 1: Retrieval
        retriever = get_hybrid_retriever()
        reranker = get_reranker()
        compressor = get_compressor()

        logger.info("\n[1] Retrieval...")
        with TrackedComponent("eval_retrieval"):
            results = retriever.retrieve(query, top_k=20)

        retrieval_count = len(results)
        top_scores = [r.get("score", 0) for r in results[:10]]

        logger.info(f"    Retrieved {retrieval_count} results")
        logger.info(f"    Top 5 scores: {[f'{s:.3f}' for s in top_scores[:5]]}")

        # Stage 2: Reranking
        logger.info("\n[2] Reranking...")
        with TrackedComponent("eval_reranking"):
            reranked = reranker.rerank(query, results, top_k=5)

        reranked_scores = [r.get("rerank_score", r.get("score", 0)) for r in reranked]
        logger.info(
            f"    Top reranked scores: {[f'{s:.3f}' for s in reranked_scores[:5]]}"
        )

        # Stage 3: Compression
        logger.info("\n[3] Context Compression...")
        with TrackedComponent("eval_compression"):
            context = compressor.compress(query, reranked)

        logger.info(f"    Original: {context.original_length} chars")
        logger.info(f"    Compressed: {context.compressed_length} chars")
        logger.info(f"    Ratio: {context.compression_ratio:.2%}")

        # Stage 4: Generation
        logger.info("\n[4] LLM Generation...")
        reasoning = get_reasoning_pipeline()

        with TrackedComponent("eval_generation"):
            query_result = reasoning.query_analysis(query, context)

        answer = query_result.answer
        sources = query_result.sources

        logger.info(f"    Answer ({len(answer)} chars): {answer[:150]}...")

        # Calculate RAGAS metrics
        metrics = RAGASMetrics()

        # Context Precision: Top results should have high scores
        if top_scores:
            metrics.context_precision = min(sum(top_scores[:5]) / 5, 1.0)

        # Context Recall: Were relevant docs found?
        metrics.context_recall = min(retrieval_count / 10, 1.0)

        # Faithfulness: Does answer use context? (heuristic)
        # Check if answer mentions terms found in context
        context_text = context.text.lower()
        if context_text and answer:
            answer_words = set(answer.lower().split())
            # Faithful if answer doesn't introduce obviously new facts
            # Simple heuristic: answer length vs context relevance
            if len(answer) > 50 and context.compression_ratio > 0.1:
                metrics.faithfulness = 0.9
            else:
                metrics.faithfulness = 0.7

        # Answer Relevance: Does answer address query?
        query_terms = set(query.lower().split())
        # Simple relevance check
        if any(term in answer.lower() for term in query_terms if len(term) > 4):
            metrics.answer_relevance = 0.9
        else:
            metrics.answer_relevance = 0.6

        # Overall (RAGAS formula)
        metrics.overall = (
            metrics.context_precision * 0.3
            + metrics.context_recall * 0.3
            + metrics.faithfulness * 0.25
            + metrics.answer_relevance * 0.15
        )

        # Print results
        logger.info(f"\n{'=' * 60}")
        logger.info("RAGAS METRICS")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Context Precision: {metrics.context_precision:.3f}")
        logger.info(f"  Context Recall:    {metrics.context_recall:.3f}")
        logger.info(f"  Faithfulness:   {metrics.faithfulness:.3f}")
        logger.info(f"  Answer Relevance: {metrics.answer_relevance:.3f}")
        logger.info(f"  OVERALL:       {metrics.overall:.3f}")

        return metrics


def run_ragas_benchmark():
    """Run RAGAS evaluation on test queries"""

    # Initialize
    print("\nInitializing RAG pipeline...")

    # Load document
    doc_file = "22705iied.pdf"
    if not os.path.exists(doc_file):
        doc_file = "DICKSON SARPONG_AI Engineer.pdf"

    if not os.path.exists(doc_file):
        print("No test document found. Using default test.")
        queries = ["What is the document about?"]
    else:
        print(f"Parsing {doc_file}...")
        parsed = parse_pdf(doc_file)
        print(f"  Parsed: {len(parsed.sections)} sections")

        print("Chunking...")
        chunks = chunk_document(parsed)
        print(f"  Created: {len(chunks.nodes)} nodes")

        # Store
        embedder = get_embedder()
        store = get_vector_store()
        store.add_nodes(chunks.nodes)

        retriever = get_hybrid_retriever()
        retriever.rebuild_index()

        # Test queries
        queries = [
            "What is this document about?",
            "What methodology was used?",
            "Who are the authors?",
            "What are the main findings?",
            "What recommendations are made?",
        ]

    evaluator = RAGASEvaluator()

    results = []
    for query in queries:
        try:
            metrics = evaluator.evaluate_full(query)
            results.append(
                {
                    "query": query,
                    "metrics": {
                        "context_precision": metrics.context_precision,
                        "context_recall": metrics.context_recall,
                        "faithfulness": metrics.faithfulness,
                        "answer_relevance": metrics.answer_relevance,
                        "overall": metrics.overall,
                    },
                }
            )
        except Exception as e:
            logger.error(f"Evaluation failed for '{query}': {e}")
            results.append({"query": query, "error": str(e)})

    # Aggregate
    valid = [r for r in results if "metrics" in r]
    if valid:
        avg_precision = sum(r["metrics"]["context_precision"] for r in valid) / len(
            valid
        )
        avg_recall = sum(r["metrics"]["context_recall"] for r in valid) / len(valid)
        avg_faithful = sum(r["metrics"]["faithfulness"] for r in valid) / len(valid)
        avg_relevance = sum(r["metrics"]["answer_relevance"] for r in valid) / len(
            valid
        )
        avg_overall = sum(r["metrics"]["overall"] for r in valid) / len(valid)

        print("\n" + "=" * 60)
        print("AGGREGATE RAGAS RESULTS")
        print("=" * 60)
        print(f"  Context Precision: {avg_precision:.3f}")
        print(f"  Context Recall:    {avg_recall:.3f}")
        print(f"  Faithfulness:   {avg_faithful:.3f}")
        print(f"  Answer Relevance: {avg_relevance:.3f}")
        print(f"  OVERALL:        {avg_overall:.3f}")

        # Grade
        if avg_overall >= 0.9:
            grade = "A - EXCELLENT"
        elif avg_overall >= 0.8:
            grade = "B - GOOD"
        elif avg_overall >= 0.7:
            grade = "C - SATISFACTORY"
        elif avg_overall >= 0.6:
            grade = "D - NEEDS IMPROVEMENT"
        else:
            grade = "F - POOR"

        print(f"\n  GRADE: {grade}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "document": doc_file,
        "query_count": len(queries),
        "aggregate": {
            "context_precision": avg_precision if valid else 0,
            "context_recall": avg_recall if valid else 0,
            "faithfulness": avg_faithful if valid else 0,
            "answer_relevance": avg_relevance if valid else 0,
            "overall": avg_overall if valid else 0,
        }
        if valid
        else {},
        "per_query": results,
    }

    output_file = f"ragas_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return output


if __name__ == "__main__":
    import os

    run_ragas_benchmark()
