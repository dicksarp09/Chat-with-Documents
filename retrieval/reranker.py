import logging
from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import CrossEncoder

from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self, model_name: str = None, top_k: int = None, min_score: float = None
    ):
        self.model_name = model_name or settings.reranker_model
        self.top_k = top_k if top_k is not None else settings.rerank_top_k
        self.min_score = (
            min_score if min_score is not None else settings.min_rerank_score
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading reranker model: {self.model_name} on {self.device}")
        self.model = CrossEncoder(self.model_name, max_length=512, device=self.device)
        logger.info(f"Reranker ready: top_k={self.top_k}, min_score={self.min_score}")

    def _get_section_title(self, result: Dict[str, Any]) -> str:
        if result.get("node") and hasattr(result["node"], "section_title"):
            return result["node"].section_title or "unknown"
        return "unknown"

    def _select_diverse_chunks(
        self, reranked: List[Dict[str, Any]], k: int
    ) -> List[Dict[str, Any]]:
        selected = []
        seen_sections = set()

        for r in reranked:
            section = self._get_section_title(r)

            if section not in seen_sections or len(selected) < k // 2:
                selected.append(r)
                seen_sections.add(section)

            if len(selected) >= k:
                break

        if len(selected) < k:
            for r in reranked:
                if r not in selected:
                    selected.append(r)
                if len(selected) >= k:
                    break

        return selected[:k]

    def rerank(
        self, query: str, results: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        k = top_k if top_k is not None else self.top_k

        if not results:
            return []

        if len(results) == 1:
            results[0]["rerank_score"] = 1.0
            return results

        logger.info(f"Reranking {len(results)} results with query: '{query[:50]}...'")

        doc_texts = []
        valid_results = []

        for r in results:
            if r.get("node") is not None:
                text = r["node"].text if hasattr(r["node"], "text") else str(r["node"])
                doc_texts.append(text)
                valid_results.append(r)
            else:
                doc_texts.append(str(r.get("text", "")))
                valid_results.append(r)

        pairs = [[query, doc] for doc in doc_texts]

        try:
            scores = self.model.predict(
                pairs, show_progress_bar=False, convert_to_numpy=True
            )
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            for r in valid_results:
                r["rerank_score"] = r.get("score", 0.0)
            return valid_results[:k]

        for i, r in enumerate(valid_results):
            r["rerank_score"] = float(scores[i])

        valid_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Score filtering only if threshold is reasonable (not overly aggressive)
        if self.min_score > -10:  # Only apply if threshold is set reasonably
            high_quality = [
                r for r in valid_results if r.get("rerank_score", 0) > self.min_score
            ]
            if high_quality:
                logger.info(
                    f"Reranker: keeping {len(high_quality)}/{len(valid_results)} above threshold {self.min_score}"
                )
                valid_results = high_quality

        reranked_diverse = self._select_diverse_chunks(valid_results, k)

        for i, r in enumerate(reranked_diverse):
            r["rerank_rank"] = i + 1

        logger.info(
            f"Reranked results, top score: {reranked_diverse[0]['rerank_score'] if reranked_diverse else 0}"
        )

        return reranked_diverse

    def rerank_with_scores(
        self, query: str, results: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        reranked = self.rerank(query, results, top_k)
        scores = [r["rerank_score"] for r in reranked]
        return reranked, scores


_reranker_instance: Optional[Reranker] = None


def get_reranker() -> Reranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance
