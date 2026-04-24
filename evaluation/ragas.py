import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


def extract_source_ids(text: str) -> List[str]:
    """Extract source node IDs from answer text."""
    pattern = r"\[source:\s*([^\]]+)\]"
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches]


def calculate_context_precision(
    query: str, retrieved_chunks: List[Dict[str, Any]]
) -> float:
    """Calculate how many retrieved chunks are relevant to the query."""
    if not retrieved_chunks:
        return 0.0

    query_terms = set(query.lower().split())
    relevant_count = 0

    for chunk in retrieved_chunks:
        chunk_text = chunk.get("text", "").lower()
        chunk_terms = set(chunk_text.split())

        overlap = query_terms & chunk_terms
        if len(overlap) >= 2:
            relevant_count += 1

    return relevant_count / len(retrieved_chunks)


def calculate_faithfulness(
    answer: str, context: str
) -> float:
    """Calculate if answer is grounded in context."""
    answer_lower = answer.lower()
    context_lower = context.lower()

    answer_sentences = re.split(r"[.!?]+", answer_lower)
    grounded_sentences = 0

    for sentence in answer_sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue

        if sentence in context_lower:
            grounded_sentences += 1
        else:
            sentence_words = set(sentence.split())
            context_words = set(context_lower.split())
            overlap = sentence_words & context_words
            if len(overlap) / len(sentence_words) > 0.5:
                grounded_sentences += 1

    total_sentences = len([s for s in answer_sentences if len(s.strip()) > 10])
    return grounded_sentences / total_sentences if total_sentences > 0 else 0.0


def calculate_answer_relevance(
    answer: str, query: str
) -> float:
    """Calculate if answer addresses the query."""
    query_terms = set(query.lower().split())
    answer_terms = set(answer.lower().split())

    overlap = query_terms & answer_terms
    return len(overlap) / len(query_terms) if query_terms else 0.0


def calculate_context_recall(
    ground_truth: str, retrieved_chunks: List[Dict[str, Any]]
) -> float:
    """Calculate if retrieved chunks contain ground truth info."""
    if not ground_truth:
        return 0.0

    gt_terms = set(ground_truth.lower().split())
    covered_terms = 0

    for chunk in retrieved_chunks:
        chunk_text = chunk.get("text", "").lower()
        chunk_terms = set(chunk_text.split())
        overlap = gt_terms & chunk_terms
        covered_terms += len(overlap)

    return min(covered_terms / len(gt_terms), 1.0) if gt_terms else 0.0


class RAGASEvaluator:
    """Simplified RAGAS metrics calculator."""

    def __init__(self):
        self.results = []

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        context: str = "",
    ) -> Dict[str, Any]:
        """Calculate RAGAS-style metrics."""

        context_precision = calculate_context_precision(query, retrieved_chunks)
        faithfulness = calculate_faithfulness(answer, context)
        answer_relevance = calculate_answer_relevance(answer, query)

        overall = (context_precision + faithfulness + answer_relevance) / 3

        result = {
            "context_precision": context_precision,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "overall": overall,
        }

        self.results.append(result)
        logger.info(f"RAGAS: precision={context_precision:.2f}, faith={faithfulness:.2f}, rel={answer_relevance:.2f}")

        return result

    def get_average_metrics(self) -> Dict[str, float]:
        if not self.results:
            return {
                "context_precision": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "overall": 0.0,
            }

        n = len(self.results)
        return {
            "context_precision": sum(r["context_precision"] for r in self.results) / n,
            "faithfulness": sum(r["faithfulness"] for r in self.results) / n,
            "answer_relevance": sum(r["answer_relevance"] for r in self.results) / n,
            "overall": sum(r["overall"] for r in self.results) / n,
        }


_evaluator: Optional[RAGASEvaluator] = None


def get_ragas_evaluator() -> RAGASEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGASEvaluator()
    return _evaluator