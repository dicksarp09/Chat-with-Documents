import sys
sys.path.insert(0, ".")

import logging
import time
import json
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def quick_benchmark():
    from app.engines.document_engine.wrapper import get_document_engine
    from storage.sqlite_store import get_sqlite_store

    print("=" * 60)
    print("QUICK BENCHMARK - Using Document Engine")
    print("=" * 60)

    # Initialize engine (loads from SQLite)
    engine = get_document_engine()

    # Get doc_id from SQLite
    sqlite = get_sqlite_store()
    datasets = sqlite.get_all_datasets()

    if not datasets:
        print("No datasets. Processing PDF...")
        with open('22705iied.pdf', 'rb') as f:
            result = engine.process(f.read(), '22705iied.pdf')
            doc_id = result['doc_id']
    else:
        doc_id = datasets[0]['id']

    print(f"Testing with doc_id: {doc_id}")

    queries = [
        "Who is at risk from food security issues?",
        "What causes food insecurity?",
        "What recommendations are made?",
        "How does climate affect food?",
        "Which regions are most affected?",
    ]

    results = []
    start = time.time()

    for q in queries:
        print(f"\nQ: {q[:50]}...")
        t0 = time.time()
        try:
            result = engine.query(q, doc_id)
            elapsed = (time.time() - t0) * 1000
            print(f"  {elapsed:.0f}ms | Answer: {result.answer[:100]}...")
            results.append({
                "query": q,
                "latency_ms": elapsed,
                "answer_len": len(result.answer),
                "sources": len(result.sources)
            })
        except Exception as e:
            print(f"  ERROR: {e}")

    total = (time.time() - start) * 1000
    avg = sum(r["latency_ms"] for r in results) / len(results) if results else 0

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Queries: {len(results)}/{len(queries)}")
    print(f"Avg latency: {avg:.0f}ms")
    print(f"Total: {total:.0f}ms")

    # Save
    with open(f"quick_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump({"results": results, "avg_ms": avg, "total_ms": total}, f, indent=2)

if __name__ == "__main__":
    quick_benchmark()