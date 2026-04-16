import sys

sys.path.insert(0, ".")
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

from compression.compressor import CompressedContext
from parsers.pdf_parser import parse_pdf
from chunking.hierarchical_chunker import chunk_document
from embeddings.embedder import get_embedder
from storage.vector_store import get_vector_store
from retrieval.hybrid_retriever import get_hybrid_retriever
from retrieval.reranker import get_reranker
from compression.compressor import get_compressor

print("Setting up pipeline...")

# Parse
parsed = parse_pdf("DICKSON SARPONG_AI Engineer.pdf")
print(f"Parsed: {len(parsed.sections)} sections")

# Chunk
structure = chunk_document(parsed)
print(f"Chunked: {len(structure.nodes)} nodes")

# Store
embedder = get_embedder()
store = get_vector_store()
store.add_nodes(structure.nodes)
print(f"Stored: {store.get_stats()['total_nodes']} nodes")

# Query
retriever = get_hybrid_retriever()
reranker = get_reranker()
compressor = get_compressor()

question = "Who is Dickson Sarpong?"
results = retriever.retrieve(question, top_k=20)
print(f"Retrieved: {len(results)} results")

reranked = reranker.rerank(question, results, top_k=5)
print(f"Reranked: {len(reranked)} results")

ctx = compressor.compress(question, reranked)
print(f"Compressed: {len(ctx.text)} chars")

# Now test the LLM Judge
from evaluation.evaluator import LLMJudge

judge = LLMJudge()

print("\nTesting LLM Judge with actual context...")

# Test with simple context first
print("Testing with simple context...")
result = judge.evaluate(
    question="Who is Dickson Sarpong?",
    context="Dickson Sarpong is an AI Engineer.",
    answer="Dickson Sarpong is an AI Engineer.",
)
print(
    f"Simple context - Result: faithful={result.faithful}, score={result.overall_score}"
)

# Test with actual context
print("\nTesting with actual context...")
try:
    result = judge.evaluate(
        question=question,
        context=ctx.text,
        answer="Dickson Sarpong is an AI Engineer with experience in building automated repository analysis systems.",
    )
    print(
        f"Actual context - Result: faithful={result.faithful}, score={result.overall_score}"
    )
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
