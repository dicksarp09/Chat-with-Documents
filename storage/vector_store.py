import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import json
from pathlib import Path

from chunking.hierarchical_chunker import ChunkNode, DocumentStructure
from embeddings.embedder import get_embedder
from core.cache import get_embedding_cache, get_retrieval_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.id_counter = 0

        self._init_index()

    def _init_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Initialized FAISS index with dimension {self.dimension}")

    def _get_next_idx(self) -> int:
        idx = self.id_counter
        self.id_counter += 1
        return idx

    def add_nodes(self, nodes: List[ChunkNode]) -> None:
        if not nodes:
            return

        # Filter out nodes that already exist
        new_nodes = [n for n in nodes if n.id not in self.id_to_idx]
        if not new_nodes:
            logger.info(f"Vector store: all {len(nodes)} nodes already exist, skipping")
            return

        logger.info(f"Vector store: adding {len(new_nodes)} new nodes ({len(nodes)} total, {len(new_nodes)} new)")

        embedder = get_embedder()

        texts = [node.text for node in new_nodes]
        embeddings = embedder.encode(texts, show_progress=True, batch_size=64)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        embeddings = embeddings.astype("float32")

        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            logger.warning("Found NaN or Inf in embeddings, replacing with zeros")
            embeddings = np.nan_to_num(embeddings)

        for i, node in enumerate(new_nodes):
            if node.id in self.id_to_idx:
                continue

            doc_vector = embeddings[i].reshape(1, -1)
            idx = self._get_next_idx()

            self.index.add(doc_vector)

            self.id_to_idx[node.id] = idx
            self.documents[node.id] = {
                "node": node,
                "embedding": embeddings[i],
                "idx": idx,
            }

        logger.info(f"Added {len(new_nodes)} nodes to vector store")

    def query_dense(
        self, query: str, top_k: int = 10, doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        embedder = get_embedder()

        # Check embedding cache first
        cache = get_embedding_cache()
        query_embedding = cache.get(query)

        if query_embedding is None:
            query_embedding = embedder.encode([query])[0]
            cache.set(query, query_embedding)
        else:
            logger.info(f"Using cached embedding for query")

        query_embedding = query_embedding.reshape(1, -1).astype("float32")

        if self.index.ntotal == 0:
            logger.warning("query_dense: index is empty")
            return []

        distances, indices = self.index.search(query_embedding, top_k)

        logger.info(
            f"query_dense: index.ntotal={self.index.ntotal}, id_counter={self.id_counter}, indices={indices[0]}"
        )

        results = []

        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= self.index.ntotal:
                logger.info(f"  Skipping idx={idx} (out of range)")
                continue

            for node_id, doc_info in self.documents.items():
                if doc_info["idx"] == idx:
                    node_doc_id = doc_info["node"].doc_id
                    if doc_id and node_doc_id != doc_id:
                        logger.info(
                            f"  Skipping idx={idx}, node_doc_id={node_doc_id} != query_doc_id={doc_id}"
                        )
                        continue

                    results.append(
                        {
                            "node_id": node_id,
                            "node": doc_info["node"],
                            "score": float(dist),
                            "rank": i + 1,
                        }
                    )
                    break
            else:
                logger.info(f"  idx={idx} not found in documents")

        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"query_dense: doc_id={doc_id}, found {len(results)} results")
        if results:
            logger.info(f"  First result doc_id: {results[0]['node'].doc_id}")

        return results[:top_k]

    def query_by_ids(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        results = []

        for node_id in node_ids:
            if node_id in self.documents:
                doc_info = self.documents[node_id]
                results.append(
                    {"node_id": node_id, "node": doc_info["node"], "score": 1.0}
                )

        return results

    def get_node(self, node_id: str) -> Optional[ChunkNode]:
        if node_id in self.documents:
            return self.documents[node_id]["node"]
        return None

    def get_all_nodes(self, doc_id: Optional[str] = None) -> List[ChunkNode]:
        nodes = []
        for doc_info in self.documents.values():
            node = doc_info["node"]
            if doc_id is None or node.doc_id == doc_id:
                nodes.append(node)
        return nodes

    def delete_document(self, doc_id: str) -> int:
        nodes_to_remove = [
            node_id
            for node_id, doc_info in self.documents.items()
            if doc_info["node"].doc_id == doc_id
        ]

        for node_id in nodes_to_remove:
            del self.documents[node_id]
            if node_id in self.id_to_idx:
                del self.id_to_idx[node_id]

        self._rebuild_index()

        logger.info(f"Deleted {len(nodes_to_remove)} nodes for doc_id: {doc_id}")
        return len(nodes_to_remove)

    def _rebuild_index(self):
        self._init_index()

        for doc_info in self.documents.values():
            embedding = doc_info["embedding"].astype("float32").reshape(1, -1)
            self.index.add(embedding)

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        index_path = save_path / "index.faiss"
        metadata_path = save_path / "metadata.json"

        if self.index is not None:
            faiss.write_index(self.index, str(index_path))

        metadata = {
            "dimension": self.dimension,
            "id_counter": self.id_counter,
            "id_to_idx": self.id_to_idx,
            "documents": {
                k: {"node": v["node"].model_dump(), "idx": v["idx"]}
                for k, v in self.documents.items()
            },
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        logger.info(f"Saved vector store to {path}")

    def load(self, path: str) -> None:
        save_path = Path(path)

        index_path = save_path / "index.faiss"
        metadata_path = save_path / "metadata.json"

        if index_path.exists():
            self.index = faiss.read_index(str(index_path))

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.dimension = metadata.get("dimension", self.dimension)
            self.id_counter = metadata.get("id_counter", 0)
            self.id_to_idx = metadata.get("id_to_idx", {})

            self.documents = {}
            for k, v in metadata.get("documents", {}).items():
                self.documents[k] = {"node": ChunkNode(**v["node"]), "idx": v["idx"]}

        logger.info(f"Loaded vector store from {path}")

    def get_stats(self) -> Dict[str, Any]:
        doc_ids = set()
        for doc_info in self.documents.values():
            doc_ids.add(doc_info["node"].doc_id)

        return {
            "total_nodes": len(self.documents),
            "total_documents": len(doc_ids),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
        }


_vector_store_instance: Optional[VectorStore] = None
_vector_store_dimension: Optional[int] = None


def get_vector_store() -> VectorStore:
    global _vector_store_instance, _vector_store_dimension
    if _vector_store_instance is None:
        # Use dimension from config (gemini-embedding-001 = 768)
        from core.config import settings
        _vector_store_dimension = settings.embedding_dimension
        _vector_store_instance = VectorStore(dimension=_vector_store_dimension)
        logger.info(f"VectorStore initialized with dimension: {_vector_store_dimension}")
    return _vector_store_instance
