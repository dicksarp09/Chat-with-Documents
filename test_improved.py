import sys

sys.path.insert(0, ".")

from parsers.pdf_parser import parse_pdf
from chunking.hierarchical_chunker import chunk_document
from embeddings.embedder import get_embedder
from storage.vector_store import get_vector_store
from retrieval.hybrid_retriever import get_hybrid_retriever
from retrieval.reranker import get_reranker
from compression.compressor import get_compressor
from llm.reasoning import get_reasoning_pipeline
from evaluation.evaluator import LLMJudge

import time

print("Setting up pipeline...")
time.sleep(2)

# Parse
parsed = parse_pdf("DICKSON SARPONG_AI Engineer.pdf")
structure = chunk_document(parsed)

# Store
embedder = get_embedder()
store = get_vector_store()
store.add_nodes(structure.nodes)

# Query
retriever = get_hybrid_retriever()
reranker = get_reranker()
compressor = get_compressor()
reasoning = get_reasoning_pipeline()
judge = LLMJudge()

questions = [
    "Who is Dickson Sarpong and what is his experience?",
    "What technical skills does Dickson have?",
    "What projects has Dickson worked on?",
]

print("\n" + "=" * 80)
print("TESTING IMPROVED RESPONSES")
print("=" * 80)

for q in questions:
    print(f"\nQuestion: {q}")

    results = retriever.retrieve(q, top_k=20)
    reranked = reranker.rerank(q, results, top_k=5)
    ctx = compressor.compress(q, reranked)
    query_result = reasoning.query_analysis(q, ctx)

    print(f"Answer: {query_result.answer}")
    print()

    eval_result = judge.evaluate(q, ctx.text, query_result.answer)
    print(
        f"Evaluation - Faithful: {eval_result.faithful}, Score: {eval_result.overall_score}"
    )
    print(f"Missing info: {eval_result.missing_info}")
    print(f"Issues: {eval_result.issues}")
    print("-" * 80)

    time.sleep(3)

print("\nDone!")
