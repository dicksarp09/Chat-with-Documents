import logging
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from storage.vector_store import get_vector_store
from embeddings.embedder import get_embedder
from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self, alpha: float = None, top_k: int = None, min_score_threshold: float = None
    ):
        self.alpha = alpha if alpha is not None else settings.hybrid_alpha
        self.top_k = top_k if top_k is not None else settings.retrieval_top_k
        self.min_score_threshold = (
            min_score_threshold
            if min_score_threshold is not None
            else settings.min_retrieval_score
        )

        self.vector_store = get_vector_store()
        self.embedder = get_embedder()
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_ids: List[str] = []
        self.corpus_texts: List[str] = []

        logger.info(
            f"Initialized HybridRetriever with alpha={self.alpha}, top_k={self.top_k}, min_score={self.min_score_threshold}"
        )

    def _build_bm25_index(self, doc_id: Optional[str] = None) -> None:
        all_nodes = self.vector_store.get_all_nodes(doc_id=None)

        self.corpus_ids = [node.id for node in all_nodes]
        self.corpus_texts = [node.text for node in all_nodes]

        if self.corpus_texts:
            tokenized_corpus = [text.lower().split() for text in self.corpus_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Built BM25 index with {len(self.corpus_texts)} documents")
        else:
            self.bm25 = None
            logger.warning("No documents found for BM25 indexing")

    def _sparse_retrieval(
        self, query: str, doc_id: Optional[str] = None, top_k: int = None
    ) -> List[Dict[str, Any]]:
        if self.bm25 is None or not self.corpus_texts:
            self._build_bm25_index(None)

        if self.bm25 is None:
            return []

        query_tokens = query.lower().split()

        doc_scores = self.bm25.get_scores(query_tokens)

        k = top_k if top_k else len(self.corpus_ids)
        top_indices = np.argsort(doc_scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            if idx < len(self.corpus_ids) and doc_scores[idx] > 0:
                node_id = self.corpus_ids[idx]
                node = self.vector_store.get_node(node_id)

                if node and (doc_id is None or node.doc_id == doc_id):
                    results.append(
                        {
                            "node_id": node_id,
                            "node": node,
                            "score": float(doc_scores[idx]),
                        }
                    )

        if results:
            max_score = max(r["score"] for r in results)
            if max_score > 0:
                for r in results:
                    r["score"] = r["score"] / max_score

        return results

    def _dense_retrieval(
        self, query: str, doc_id: Optional[str] = None, top_k: int = None
    ) -> List[Dict[str, Any]]:
        k = top_k if top_k else self.top_k

        results = self.vector_store.query_dense(query, top_k=k, doc_id=doc_id)

        return results

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return results

        scores = [r["score"] for r in results]
        max_score = max(scores) if max(scores) > 0 else 1

        for r in results:
            r["score"] = r["score"] / max_score

        return results

    def retrieve(
        self, query: str, doc_id: Optional[str] = None, top_k: int = None
    ) -> List[Dict[str, Any]]:
        k = top_k if top_k is not None else self.top_k

        logger.info(
            f"Starting hybrid retrieval for query: '{query[:50]}...' (doc_id: {doc_id})"
        )

        sparse_results = self._sparse_retrieval(query, doc_id, k * 2)
        logger.info(f"Sparse retrieval returned {len(sparse_results)} results")

        dense_results = self._dense_retrieval(query, doc_id, k * 2)
        logger.info(f"Dense retrieval returned {len(dense_results)} results")

        score_map: Dict[str, Dict[str, Any]] = {}

        for r in sparse_results:
            node_id = r["node_id"]
            score_map[node_id] = {
                "node_id": node_id,
                "node": r["node"],
                "sparse_score": r["score"],
                "dense_score": 0.0,
                "score": 0.0,
            }

        for r in dense_results:
            node_id = r["node_id"]
            if node_id in score_map:
                score_map[node_id]["dense_score"] = r["score"]
            else:
                score_map[node_id] = {
                    "node_id": node_id,
                    "node": r["node"],
                    "sparse_score": 0.0,
                    "dense_score": r["score"],
                    "score": 0.0,
                }

        for node_id, data in score_map.items():
            data["score"] = (
                self.alpha * data["dense_score"]
                + (1 - self.alpha) * data["sparse_score"]
            )

        # Filter by minimum score threshold (precision improvement)
        filtered = [
            r for r in score_map.values() if r["score"] >= self.min_score_threshold
        ]
        filtered.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            f"After score filtering ({self.min_score_threshold}): {len(filtered)}/{len(score_map)} results"
        )

        merged_results = filtered if filtered else list(score_map.values())
        if not filtered:
            merged_results.sort(key=lambda x: x["score"], reverse=True)

        diverse_results = self._ensure_diversity(merged_results, k)

        logger.info(
            f"Hybrid retrieval merged to {len(diverse_results)} diverse results"
        )

        return diverse_results

    def _ensure_diversity(
        self, results: List[Dict[str, Any]], k: int
    ) -> List[Dict[str, Any]]:
        if len(results) <= k:
            return results

        selected = []
        seen_sections = set()

        for r in results:
            if len(selected) >= k:
                break

            node = r.get("node")
            if node and hasattr(node, "section_title"):
                section = node.section_title or "unknown"
            else:
                section = "unknown"

            if section not in seen_sections or len(selected) < k // 2:
                selected.append(r)
                seen_sections.add(section)

        if len(selected) < k:
            for r in results:
                if r not in selected:
                    selected.append(r)
                if len(selected) >= k:
                    break

        return selected

    def rebuild_index(self, doc_id: Optional[str] = None) -> None:
        logger.info(f"rebuild_index called with doc_id={doc_id}")
        self._build_bm25_index(doc_id)

    def get_retrieval_stats(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "top_k": self.top_k,
            "corpus_size": len(self.corpus_ids),
            "bm25_ready": self.bm25 is not None,
        }


_hybrid_retriever_instance: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever_instance
    if _hybrid_retriever_instance is None:
        _hybrid_retriever_instance = HybridRetriever()
    return _hybrid_retriever_instance
