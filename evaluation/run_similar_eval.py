import sys
sys.path.insert(0, ".")

import logging
import time
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_evaluation():
    from app.engines.document_engine.wrapper import get_document_engine

    engine = get_document_engine()

    print("Processing PDF...")
    with open('22705iied.pdf', 'rb') as f:
        result = engine.process(f.read(), '22705iied.pdf')
        doc_id = result['doc_id']
    print(f"Processed: {doc_id}")

    questions = [
        "What is this research paper about?",
        "What are the main findings or conclusions?",
        "What methodology was used in this research?",
        "Who are the authors of this paper?",
        "What are the key recommendations?"
    ]

    results = []
    total_start = time.time()

    for q in questions:
        print(f"\nQ: {q}")
        t0 = time.time()
        answer = engine.query(q, doc_id)
        elapsed = (time.time() - t0) * 1000

        results.append({
            "question": q,
            "answer": answer.answer,
            "sources": answer.sources,
            "latency_ms": elapsed
        })
        print(f"  Latency: {elapsed:.0f}ms")
        print(f"  Answer: {answer.answer[:150]}...")

    total_time = (time.time() - total_start) * 1000

    output = {
        "timestamp": datetime.now().isoformat(),
        "document": "22705iied.pdf (Research Paper)",
        "total_time_ms": total_time,
        "results": results
    }

    fname = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {fname}")

if __name__ == "__main__":
    run_evaluation()