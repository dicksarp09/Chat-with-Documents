import sys
sys.path.insert(0, ".")

import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def main():
    from app.engines.document_engine.wrapper import get_document_engine
    from storage.sqlite_store import get_sqlite_store
    from embeddings.embedder import warmup_embedder

    print("=" * 60)
    print("IMPROVED RAG PIPELINE BENCHMARK")
    print("=" * 60)

    print("\n[1] Warming up embedder...")
    t0 = time.time()
    warmup_embedder()
    warmup_time = (time.time() - t0) * 1000
    print(f"    Warmup: {warmup_time:.0f}ms")

    print("\n[2] Loading document engine...")
    t0 = time.time()
    engine = get_document_engine()
    engine_time = (time.time() - t0) * 1000
    print(f"    Engine load: {engine_time:.0f}ms")

    sqlite = get_sqlite_store()
    datasets = sqlite.get_all_datasets()
    doc_id = datasets[0]['id'] if datasets else None
    print(f"    Doc ID: {doc_id}")

    if not doc_id:
        print("    Processing PDF first...")
        with open('22705iied.pdf', 'rb') as f:
            result = engine.process(f.read(), '22705iied.pdf')
            doc_id = result['doc_id']
    else:
        print(f"    Loading existing dataset...")
        # Process again to load vectors into vector store
        with open('22705iied.pdf', 'rb') as f:
            result = engine.process(f.read(), '22705iied.pdf')
            doc_id = result['doc_id']

    queries = [
        "Who is at risk from food security issues?",
        "What are the main causes of food insecurity?",
        "What recommendations does the document make?",
        "How does climate change affect food production?",
        "What regions are most affected by food insecurity?",
    ]

    print(f"\n[3] Running {len(queries)} queries...")
    results = []
    total_start = time.time()

    for i, q in enumerate(queries):
        print(f"\n    Query {i+1}: {q[:40]}...")
        t0 = time.time()
        try:
            result = engine.query(q, doc_id)

            # Handle both dict and object returns
            if isinstance(result, dict):
                answer = result.get("answer", "")
                sources = result.get("sources", [])
            else:
                answer = result.answer
                sources = result.sources

            elapsed = (time.time() - t0) * 1000

            print(f"    Latency: {elapsed:.0f}ms")
            print(f"    Answer: {answer[:100]}...")

            results.append({
                "query": q,
                "latency_ms": elapsed,
                "answer_length": len(answer),
                "sources": len(sources)
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"query": q, "error": str(e), "latency_ms": 0})

    total_time = (time.time() - total_start) * 1000

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    successful = [r for r in results if "error" not in r]
    if successful:
        avg_latency = sum(r["latency_ms"] for r in successful) / len(successful)
        min_latency = min(r["latency_ms"] for r in successful)
        max_latency = max(r["latency_ms"] for r in successful)

        print(f"Queries run: {len(successful)}/{len(queries)}")
        print(f"Avg latency: {avg_latency:.0f}ms")
        print(f"Min latency: {min_latency:.0f}ms")
        print(f"Max latency: {max_latency:.0f}ms")
        print(f"Total time: {total_time:.0f}ms")

        output = {
            "timestamp": datetime.now().isoformat(),
            "warmup_ms": warmup_time,
            "engine_load_ms": engine_time,
            "queries": len(queries),
            "successful": len(successful),
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "total_ms": total_time,
            "per_query": results
        }

        fname = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {fname}")

if __name__ == "__main__":
    main()