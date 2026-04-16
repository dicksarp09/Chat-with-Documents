import sys

sys.path.insert(0, ".")

from storage.vector_store import get_vector_store
from embeddings.embedder import get_embedder
from retrieval.hybrid_retriever import get_hybrid_retriever

print("=== Vector Store Debug ===")
vs = get_vector_store()
stats = vs.get_stats()
print(f"Vector store stats: {stats}")

nodes = vs.get_all_nodes()
print(f"Total nodes in vector store: {len(nodes)}")

print("\n=== Query Test ===")
hr = get_hybrid_retriever()
print(f"BM25 corpus size: {len(hr.corpus_ids)}")

# Test retrieval
query = "What is this document about?"
results = hr.retrieve(query, doc_id=None, top_k=5)
print(f"Retrieved {len(results)} results")

if results:
    print("First result:")
    r = results[0]
    print(f"  Node ID: {r.get('node_id', 'N/A')}")
    print(f"  Score: {r.get('score', 'N/A')}")
    node = r.get("node")
    if node:
        print(f"  Text preview: {node.text[:200]}...")
