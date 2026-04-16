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

print("Testing improved responses...")

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

questions = [
    "Who is Dickson Sarpong and what is his experience?",
    "What technical skills does Dickson have?",
    "What projects has Dickson worked on?",
]

for q in questions:
    print(f"\nQuestion: {q}")
    results = retriever.retrieve(q, top_k=20)
    reranked = reranker.rerank(q, results, top_k=5)
    ctx = compressor.compress(q, reranked)
    query_result = reasoning.query_analysis(q, ctx)
    print(f"Answer: {query_result.answer[:500]}...")
    print()
