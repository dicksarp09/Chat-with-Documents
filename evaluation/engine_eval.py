import sys
sys.path.insert(0, ".")

import logging
import time
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def evaluate():
    from app.engines.document_engine.wrapper import get_document_engine
    from storage.sqlite_store import get_sqlite_store

    print("=" * 60)
    print("IMPROVED PIPELINE EVALUATION")
    print("=" * 60)

    engine = get_document_engine()

    sqlite = get_sqlite_store()
    datasets = sqlite.get_all_datasets()
    print(f"\nDatasets in SQLite: {len(datasets)}")

    if not datasets:
        print("No datasets! Processing PDF first...")
        with open('22705iied.pdf', 'rb') as f:
            result = engine.process(f.read(), '22705iied.pdf')
            doc_id = result['doc_id']
            print(f"Processed: {doc_id}")
    else:
        doc_id = datasets[0]['id']
        print(f"Using existing doc: {datasets[0]['name']} ({doc_id})")

    queries = [
        "Who is at risk from food security issues?",
        "What are the main causes of food insecurity?",
        "What recommendations does the document make?",
        "How does climate change affect food production?",
        "What regions are most affected by food insecurity?",
    ]

    results = []
    total_start = time.time()

    for q in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")

        query_start = time.time()

        try:
            result = engine.query(q, doc_id)
            total_time = (time.time() - query_start) * 1000

            print(f"Latency: {total_time:.0f}ms")
            print(f"Answer: {result.answer[:250]}...")
            print(f"Sources: {len(result.sources)}")

            if result.answer and result.answer != "No relevant content found" and "error" not in result.answer.lower():
                results.append({
                    "query": q,
                    "answer": result.answer,
                    "sources_count": len(result.sources),
                    "latency_ms": total_time,
                    "success": True
                })
            else:
                results.append({
                    "query": q,
                    "answer": result.answer,
                    "latency_ms": total_time,
                    "success": False
                })

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "query": q,
                "error": str(e),
                "success": False
            })

    total_eval_time = (time.time() - total_start) * 1000

    successful = [r for r in results if r.get("success")]
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"Total queries: {len(results)}")
    print(f"Successful: {len(successful)}")

    if successful:
        avg_latency = sum(r["latency_ms"] for r in successful) / len(successful)
        total_latency = sum(r["latency_ms"] for r in successful)
        print(f"Avg latency: {avg_latency:.0f}ms")
        print(f"Total latency: {total_latency:.0f}ms")

        output = {
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "total_queries": len(results),
            "successful": len(successful),
            "avg_latency_ms": avg_latency,
            "total_latency_ms": total_eval_time,
            "per_query": results
        }

        fname = f"pipeline_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {fname}")

if __name__ == "__main__":
    evaluate()