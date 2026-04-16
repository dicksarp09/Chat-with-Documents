import sys

sys.path.insert(0, ".")

from parsers.pdf_parser import parse_pdf
from chunking.hierarchical_chunker import chunk_document
from embeddings.embedder import get_embedder
from storage.vector_store import get_vector_store
from retrieval.hybrid_retriever import get_hybrid_retriever
from retrieval.reranker import get_reranker
from compression.compressor import get_compressor
from llm.groq_client import get_groq_client
from llm.reasoning import get_reasoning_pipeline

print("=== Document Intelligence Engine Test ===")
print()

# 1. Parse
parsed = parse_pdf("DICKSON SARPONG_AI Engineer.pdf")
print(f"[1] Parsed: {len(parsed.sections)} sections, {len(parsed.full_text)} chars")

# 2. Chunk
structure = chunk_document(parsed)
print(f"[2] Chunked: {len(structure.nodes)} nodes")

# 3. Embed & Store
embedder = get_embedder()
store = get_vector_store()
store.add_nodes(structure.nodes)
stats = store.get_stats()
print(f"[3] Stored: {stats['total_nodes']} nodes")

# 4. Hybrid Retrieval
retriever = get_hybrid_retriever()
results = retriever.retrieve("Who is Dickson Sarpong?", top_k=10)
print(f"[4] Retrieved: {len(results)} results")

# 5. Rerank
reranker = get_reranker()
reranked = reranker.rerank("Who is Dickson Sarpong?", results, top_k=5)
print(f"[5] Reranked top 5")

# 6. Compress
compressor = get_compressor()
ctx = compressor.compress("Who is Dickson Sarpong?", reranked)
print(f"[6] Compressed: {len(ctx.text)} chars")

# 7. Reasoning
reasoning = get_reasoning_pipeline()
query_result = reasoning.query_analysis("Who is Dickson Sarpong?", ctx)

print()
print("=== Query Result ===")
print(
    f"Answer: {query_result.answer[:300]}..."
    if len(query_result.answer) > 300
    else f"Answer: {query_result.answer}"
)
print(f"Sources: {len(query_result.sources)} nodes")
print()
print("=== TEST PASSED ===")
