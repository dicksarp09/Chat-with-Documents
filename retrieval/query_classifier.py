import logging
from enum import Enum
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    EXTRACTION = "extraction"  # "Who", "What", "Where" - specific facts
    SUMMARIZATION = "summarization"  # "Summarize", "What are the main" - overview
    REASONING = "reasoning"  # "Why", "How does", "What causes" - causal analysis
    COMPARISON = "comparison"  # "Compare", "Difference between" - comparative
    RECOMMENDATION = "recommendation"  # "What should", "Recommend", "How to" - action-oriented


class QueryIntentClassifier:
    """
    Classifies query intent to optimize retrieval strategy.
    Different intents need different retrieval behaviors.
    """

    def __init__(self):
        self.patterns = {
            QueryIntent.EXTRACTION: [
                r"^who\s", r"^what\s", r"^where\s", r"^when\s", r"^which\s",
                r"^how many", r"^how much",
                r"name\s", r"identify\s", r"list\s",
            ],
            QueryIntent.SUMMARIZATION: [
                r"summarize", r"overview", r"summary",
                r"what are the main", r"what is the", r"describe",
                r"explain the", r"outline",
            ],
            QueryIntent.REASONING: [
                r"^why\s", r"^how does", r"^how do", r"^because",
                r"what causes", r"what leads to", r"result in",
                r"impact of", r"effect of", r"relationship between",
            ],
            QueryIntent.COMPARISON: [
                r"compare", r"difference between", r"similar to",
                r"versus", r"vs\s", r"contrast",
            ],
            QueryIntent.RECOMMENDATION: [
                r"recommend", r"should\s", r"what should",
                r"how to", r"steps to", r"solution",
                r"best way", r"advice",
            ],
        }

    def classify(self, query: str) -> QueryIntent:
        """Classify query intent from text."""
        query_lower = query.lower().strip()

        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.info(f"Query intent: {intent.value} (matched: {pattern})")
                    return intent

        if "?" in query_lower:
            return QueryIntent.EXTRACTION

        return QueryIntent.SUMMARIZATION

    def get_retrieval_config(self, intent: QueryIntent) -> Dict[str, Any]:
        """Get retrieval configuration based on intent."""

        configs = {
            QueryIntent.EXTRACTION: {
                "top_k": 10,
                "use_bm25_weight": 0.6,  # Prioritize keyword matching
                "use_dense_weight": 0.4,
                "require_exact_match": True,
                "expand_query": True,
            },
            QueryIntent.SUMMARIZATION: {
                "top_k": 15,
                "use_bm25_weight": 0.4,
                "use_dense_weight": 0.6,  # Prioritize semantic similarity
                "require_exact_match": False,
                "expand_query": True,
            },
            QueryIntent.REASONING: {
                "top_k": 20,
                "use_bm25_weight": 0.3,
                "use_dense_weight": 0.7,  # Need contextual understanding
                "require_exact_match": False,
                "expand_query": True,
                "include_related": True,
            },
            QueryIntent.COMPARISON: {
                "top_k": 15,
                "use_bm25_weight": 0.5,
                "use_dense_weight": 0.5,
                "require_exact_match": False,
                "expand_query": True,
            },
            QueryIntent.RECOMMENDATION: {
                "top_k": 15,
                "use_bm25_weight": 0.3,
                "use_dense_weight": 0.7,
                "require_exact_match": False,
                "expand_query": True,
                "filter_action_words": True,
            },
        }

        return configs.get(intent, configs[QueryIntent.SUMMARIZATION])

    def expand_query(self, query: str, intent: QueryIntent) -> List[str]:
        """Expand query with related terms based on intent."""

        base_query = query.lower().strip()

        expansions = {
            QueryIntent.EXTRACTION: [base_query],
            QueryIntent.SUMMARIZATION: [
                base_query,
                f"main {base_query}",
                f"key {base_query}",
                f"important {base_query}",
            ],
            QueryIntent.REASONING: [
                base_query,
                f"causes of {base_query}",
                f"factors {base_query}",
                f"why {base_query}",
            ],
            QueryIntent.COMPARISON: [
                base_query,
            ],
            QueryIntent.RECOMMENDATION: [
                base_query,
                f"best {base_query}",
                f"solution {base_query}",
                f"approach {base_query}",
            ],
        }

        expanded = expansions.get(intent, [base_query])
        logger.info(f"Query expanded: {expanded}")
        return expanded


_intent_classifier: Optional[QueryIntentClassifier] = None


def get_intent_classifier() -> QueryIntentClassifier:
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = QueryIntentClassifier()
    return _intent_classifier