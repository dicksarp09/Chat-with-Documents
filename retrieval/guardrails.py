import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    min_score: float = 0.1
    min_relevance_score: float = 0.2
    fallback_answer: str = (
        "I couldn't find relevant information in the document to answer your question."
    )
    evidence_required: bool = True
    min_evidence_sources: int = 1
    low_quality_answer_prefix: str = (
        "[Note: This is a low-confidence answer based on limited context.]\n\n"
    )


@dataclass
class RetrievalResult:
    results: List[Dict[str, Any]]
    has_results: bool = False
    quality_score: float = 0.0
    meets_threshold: bool = False
    has_evidence: bool = False
    fallback_triggered: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_results": self.has_results,
            "quality_score": round(self.quality_score, 3),
            "meets_threshold": self.meets_threshold,
            "has_evidence": self.has_evidence,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
            "result_count": len(self.results),
        }


class RetrievalGuardrails:
    def __init__(self, config: RetrievalConfig = None):
        self.config = config or RetrievalConfig()

    def apply(self, results: List[Dict[str, Any]], query: str = "") -> RetrievalResult:
        if not results:
            logger.warning(f"No retrieval results for query: {query[:50]}...")
            return RetrievalResult(
                results=[],
                has_results=False,
                quality_score=0.0,
                meets_threshold=False,
                has_evidence=False,
                fallback_triggered=True,
                fallback_reason="no_results",
            )

        filtered = [r for r in results if r.get("score", 0) >= self.config.min_score]

        if not filtered:
            logger.warning(
                f"Results filtered out due to low scores. Top score: {results[0].get('score', 0):.3f}"
            )
            return RetrievalResult(
                results=results[:3],
                has_results=True,
                quality_score=results[0].get("score", 0),
                meets_threshold=False,
                has_evidence=False,
                fallback_triggered=True,
                fallback_reason="below_min_score",
            )

        top_score = filtered[0].get("score", 0)
        meets_threshold = top_score >= self.config.min_relevance_score

        has_evidence = self._verify_evidence(filtered)

        if not has_evidence:
            logger.warning("Results lack sufficient evidence sources")
            return RetrievalResult(
                results=filtered,
                has_results=True,
                quality_score=top_score,
                meets_threshold=meets_threshold,
                has_evidence=False,
                fallback_triggered=True,
                fallback_reason="no_evidence",
            )

        if not meets_threshold:
            logger.info(f"Low confidence results: top_score={top_score:.3f}")
            return RetrievalResult(
                results=filtered,
                has_results=True,
                quality_score=top_score,
                meets_threshold=False,
                has_evidence=True,
                fallback_triggered=False,
                fallback_reason="low_confidence",
            )

        logger.info(
            f"Good retrieval: {len(filtered)} results, top_score={top_score:.3f}"
        )
        return RetrievalResult(
            results=filtered,
            has_results=True,
            quality_score=top_score,
            meets_threshold=meets_threshold,
            has_evidence=True,
            fallback_triggered=False,
            fallback_reason="",
        )

    def _verify_evidence(self, results: List[Dict[str, Any]]) -> bool:
        source_ids = set()
        for r in results:
            node_id = r.get("node_id") or (r.get("node") and r["node"].get("id"))
            if node_id:
                source_ids.add(node_id)
        return len(source_ids) >= self.config.min_evidence_sources

    def get_answer_with_fallback(
        self,
        result: RetrievalResult,
        llm_answer: str,
        query: str,
    ) -> tuple[str, bool]:
        if result.fallback_triggered and not result.has_results:
            logger.info(f"Using full fallback for query: {query[:50]}...")
            return self.config.fallback_answer, True

        if result.fallback_triggered and not result.meets_threshold:
            logger.info(f"Adding low-confidence warning for query: {query[:50]}...")
            return self.config.low_quality_answer_prefix + llm_answer, True

        if result.has_results and not llm_answer:
            logger.warning(f"LLM returned empty answer, using fallback")
            return self.config.fallback_answer, True

        return llm_answer, result.fallback_triggered

    def should_retry(self, result: RetrievalResult) -> bool:
        return not result.has_results or result.fallback_reason == "no_results"

    def get_retry_query(self, original_query: str, attempt: int) -> str:
        expanded = f"{original_query} (search for relevant information)"
        return expanded


class MultiStageGuardrails:
    """Guardrails for different retrieval stages"""

    def __init__(self):
        self.hybrid_config = RetrievalConfig(min_score=0.1, min_relevance_score=0.2)
        self.rerank_config = RetrievalConfig(min_score=0.2, min_relevance_score=0.4)
        self.final_config = RetrievalConfig(min_score=0.3, min_relevance_score=0.5)

        self.hybrid_guardrails = RetrievalGuardrails(self.hybrid_config)
        self.rerank_guardrails = RetrievalGuardrails(self.rerank_config)
        self.final_guardrails = RetrievalGuardrails(self.final_config)

    def check_hybrid(self, results: List[Dict]) -> RetrievalResult:
        return self.hybrid_guardrails.apply(results)

    def check_rerank(self, results: List[Dict]) -> RetrievalResult:
        return self.rerank_guardrails.apply(results)

    def check_final(self, results: List[Dict]) -> RetrievalResult:
        return self.final_guardrails.apply(results)


def create_default_guardrails() -> RetrievalGuardrails:
    return RetrievalGuardrails(RetrievalConfig())


def validate_retrieval_response(response: Dict[str, Any]) -> bool:
    required_fields = ["answer"]
    for field in required_fields:
        if field not in response:
            logger.error(f"Missing required field: {field}")
            return False
    return True


def create_safe_response(
    result: RetrievalResult,
    llm_answer: str,
    query: str,
    sources: List[str] = None,
) -> Dict[str, Any]:
    guardrails = create_default_guardrails()
    answer, used_fallback = guardrails.get_answer_with_fallback(
        result, llm_answer, query
    )

    return {
        "answer": answer,
        "sources": sources or [],
        "metadata": {
            "fallback_used": used_fallback,
            "quality_score": result.quality_score,
            "has_evidence": result.has_evidence,
            "result_count": len(result.results),
            "fallback_reason": result.fallback_reason,
        },
    }
