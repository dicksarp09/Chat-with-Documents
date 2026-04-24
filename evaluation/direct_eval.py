import sys
sys.path.insert(0, ".")

import logging
import time
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def evaluate():
    from storage.sqlite_store import get_sqlite_store
    from storage.vector_store import get_vector_store
    from retrieval.hybrid_retriever import get_hybrid_retriever
    from retrieval.reranker import get_reranker
    from retrieval.query_classifier import get_intent_classifier
    from compression.compressor import get_compressor
    from llm.reasoning import get_reasoning_pipeline

    sqlite = get_sqlite_store()
    datasets = sqlite.get_all_datasets()
    print(f"\nDatasets found: {len(datasets)}")

    if not datasets:
        print("No datasets found!")
        return

    doc_id = datasets[0]['id']
    print(f"Using doc_id: {doc_id}")

    queries = [
        ("Who is at risk from food security issues?", "extraction"),
        ("What are the main causes of food insecurity?", "reasoning"),
        ("What recommendations does the document make?", "recommendation"),
        ("How does climate change affect food production?", "reasoning"),
        ("What regions are most affected by food insecurity?", "extraction"),
    ]

    classifier = get_intent_classifier()
    retriever = get_hybrid_retriever()
    reranker = get_reranker()
    compressor = get_compressor()
    reasoning = get_reasoning_pipeline()

    results = []
    total_start = time.time()

    for q, expected_intent in queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"Expected intent: {expected_intent}")

        intent = classifier.classify(q)
        print(f"Detected intent: {intent.value}")

        query_start = time.time()

        try:
            # Retrieval
            t0 = time.time()
            retrieved = retriever.retrieve(q, doc_id=doc_id, top_k=20)
            retrieval_time = (time.time() - t0) * 1000
            print(f"Retrieved: {len(retrieved)} in {retrieval_time:.0f}ms")

            if not retrieved:
                print("No results retrieved!")
                continue

            # Show top results
            top_scores = [r.get('score', 0) for r in retrieved[:5]]
            print(f"Top scores: {[f'{s:.3f}' for s in top_scores]}")

            # Reranking
            t0 = time.time()
            reranked = reranker.rerank(q, retrieved, top_k=5)
            rerank_time = (time.time() - t0) * 1000
            print(f"Reranked: {len(reranked)} in {rerank_time:.0f}ms")

            # Compression
            t0 = time.time()
            context = compressor.compress(q, reranked)
            compression_time = (time.time() - t0) * 1000
            print(f"Compressed: {context.compressed_length}/{context.original_length} chars in {compression_time:.0f}ms")
            print(f"Context preview: {context.text[:200]}...")

            # Generation
            t0 = time.time()
            result = reasoning.query_analysis(q, context)
            gen_time = (time.time() - t0) * 1000
            total_time = (time.time() - query_start) * 1000

            print(f"Generated in {gen_time:.0f}ms (total: {total_time:.0f}ms)")
            print(f"Answer: {result.answer[:300]}...")
            print(f"Sources: {len(result.sources)}")

            # Calculate metrics
            precision = sum(r.get('score', 0) for r in retrieved[:10]) / 10 if retrieved else 0
            faithfulness = _calc_faithfulness(result.answer, context.text)
            relevance = _calc_relevance(q, result.answer)
            groundedness = _calc_groundedness(result.answer, context.text)

            results.append({
                "query": q,
                "intent": intent.value,
                "expected_intent": expected_intent,
                "precision": precision,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "groundedness": groundedness,
                "latency_ms": total_time
            })

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    total_eval_time = (time.time() - total_start) * 1000

    if results:
        avg_p = sum(r["precision"] for r in results) / len(results)
        avg_f = sum(r["faithfulness"] for r in results) / len(results)
        avg_r = sum(r["relevance"] for r in results) / len(results)
        avg_g = sum(r["groundedness"] for r in results) / len(results)
        avg_lat = sum(r["latency_ms"] for r in results) / len(results)

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Precision:    {avg_p:.3f}")
        print(f"Faithfulness: {avg_f:.3f}")
        print(f"Relevance:    {avg_r:.3f}")
        print(f"Groundedness: {avg_g:.3f}")
        print(f"Avg Latency:  {avg_lat:.0f}ms")
        print(f"Total Eval:   {total_eval_time:.0f}ms")

        overall = (avg_p + avg_f + avg_r + avg_g) / 4
        if overall >= 0.9: grade = "A - EXCELLENT"
        elif overall >= 0.8: grade = "B - GOOD"
        elif overall >= 0.7: grade = "C - SATISFACTORY"
        elif overall >= 0.6: grade = "D - NEEDS IMPROVEMENT"
        else: grade = "F - POOR"
        print(f"OVERALL: {overall:.3f} ({grade})")

        output = {
            "timestamp": datetime.now().isoformat(),
            "aggregate": {
                "precision": avg_p, "faithfulness": avg_f,
                "relevance": avg_r, "groundedness": avg_g,
                "avg_latency_ms": avg_lat, "total_latency_ms": total_eval_time
            },
            "per_query": results
        }
        with open(f"direct_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(output, f, indent=2)

def _calc_faithfulness(answer, context):
    if not answer or not context: return 0
    words = set(w.lower() for w in answer.split() if len(w) > 4)
    ctx_words = set(context.lower().split())
    if not words: return 1.0
    return sum(1 for w in words if w in ctx_words) / len(words)

def _calc_relevance(query, answer):
    if not query or not answer: return 0
    terms = set(w.lower() for w in query.split() if len(w) > 3)
    ans = answer.lower()
    return sum(1 for t in terms if t in ans) / len(terms) if terms else 0

def _calc_groundedness(answer, context):
    if not answer or not context: return 0.5
    sents = [s.strip() for s in answer.split('.') if s.strip()]
    if not sents: return 1.0
    supported = sum(1 for s in sents if any(w in context.lower() for w in s.split() if len(w) > 3))
    return supported / len(sents)

if __name__ == "__main__":
    evaluate()