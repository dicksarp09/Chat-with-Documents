import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from llm.groq_client import get_groq_client
from compression.compressor import CompressedContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LLM_JUDGE_PROMPT = """You are an expert evaluator for RAG systems. Judge answer quality on:

1. FAITHFULNESS: Is answer grounded in context? Any hallucinations?
2. RELEVANCE: Does it answer what was asked? No irrelevant details?
3. CONCISENESS: Is it appropriately detailed (not too short/long)?
4. GROUNDEDNESS: Are claims backed by evidence? Are evidence node IDs valid?
5. INFORMATION DENSITY: Does it include specific details (technologies, metrics, outcomes)?

EVALUATION RULES:
- If answer includes irrelevant details → reduce relevance
- If answer is too verbose for question type → penalize
- If factual question answered with essay → severe penalty
- If broad question answered with 1 word → partial penalty
- If claims lack evidence support → reduce groundedness
- If answer is too vague (no specifics) → reduce information density
- "Not found in context" is CORRECT when info missing

GOOD answer characteristics:
- Includes specific technologies, metrics, or outcomes
- Claims are backed by context
- Concise but informative
- Matches question type (factual=short, specific=detailed, broad=summary)

Output ONLY this JSON:
{{
    "faithful": true/false,
    "relevant": true/false,
    "concise": true/false,
    "grounded": true/false,
    "specific": true/false,
    "hallucinations": [],
    "missing_info": [],
    "issues": [],
    "faithfulness_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "concision_score": 0.0-1.0,
    "groundedness_score": 0.0-1.0,
    "information_density_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "explanation": "brief"
}}

Question: {question}
Context: {context}
Answer: {answer}
"""


@dataclass
class EvaluationResult:
    faithful: bool
    relevant: bool
    grounded: bool
    hallucinations: List[str]
    missing_info: List[str]
    issues: List[str]
    faithfulness_score: float
    relevance_score: float
    concision_score: float = 1.0
    groundedness_score: float = 1.0
    information_density_score: float = 1.0
    overall_score: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithful": self.faithful,
            "relevant": self.relevant,
            "grounded": self.grounded,
            "hallucinations": self.hallucinations,
            "missing_info": self.missing_info,
            "issues": self.issues,
            "faithfulness_score": self.faithfulness_score,
            "relevance_score": self.relevance_score,
            "concision_score": self.concision_score,
            "groundedness_score": self.groundedness_score,
            "information_density_score": self.information_density_score,
            "overall_score": self.overall_score,
            "explanation": self.explanation,
        }


@dataclass
class TestCase:
    question: str
    ground_truth: Optional[str] = None
    expected_key_points: Optional[List[str]] = None


class LLMJudge:
    def __init__(self):
        self.llm = get_groq_client()

    def evaluate(self, question: str, context: str, answer: str) -> EvaluationResult:
        prompt = LLM_JUDGE_PROMPT.format(
            question=question, context=context, answer=answer
        )

        try:
            logger.info(
                f"Calling LLM for evaluation with context length: {len(context)}"
            )
            response = self.llm.generate_json(prompt, max_retries=3)

            logger.info(f"Response type: {type(response)}")

            if not isinstance(response, dict):
                logger.warning(f"Response is not a dict: {type(response)}")
                raise ValueError(f"Expected dict, got {type(response)}")

            faithfulness_score = float(response.get("faithfulness_score", 0.0))
            relevance_score = float(response.get("relevance_score", 0.0))
            concision_score = float(response.get("concision_score", 1.0))
            groundedness_score = float(response.get("groundedness_score", 1.0))
            info_density_score = float(response.get("information_density_score", 1.0))

            overall_score = (
                faithfulness_score * 0.3
                + relevance_score * 0.2
                + concision_score * 0.15
                + groundedness_score * 0.2
                + info_density_score * 0.15
            )

            return EvaluationResult(
                faithful=response.get("faithful", False),
                relevant=response.get("relevant", False),
                grounded=response.get("grounded", False),
                hallucinations=response.get("hallucinations", []),
                missing_info=response.get("missing_info", []),
                issues=response.get("issues", []),
                faithfulness_score=faithfulness_score,
                relevance_score=relevance_score,
                concision_score=concision_score,
                groundedness_score=groundedness_score,
                information_density_score=info_density_score,
                overall_score=overall_score,
                explanation=str(response.get("explanation", "")),
            )
        except Exception as e:
            logger.error(f"LLM Judge evaluation failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return EvaluationResult(
                faithful=False,
                relevant=False,
                grounded=False,
                hallucinations=[str(e)],
                missing_info=[],
                issues=["Evaluation failed"],
                faithfulness_score=0.0,
                relevance_score=0.0,
                concision_score=0.0,
                groundedness_score=0.0,
                information_density_score=0.0,
                overall_score=0.0,
                explanation=f"Error during evaluation: {str(e)}",
            )


class RagasMetrics:
    def __init__(self):
        self.llm_judge = LLMJudge()

    def compute_faithfulness(self, question: str, context: str, answer: str) -> float:
        result = self.llm_judge.evaluate(question, context, answer)
        return result.faithfulness_score

    def compute_context_precision(
        self, question: str, context: str, answer: str
    ) -> float:
        result = self.llm_judge.evaluate(question, context, answer)
        return result.relevance_score

    def compute_answer_relevance(
        self, question: str, context: str, answer: str
    ) -> float:
        result = self.llm_judge.evaluate(question, context, answer)
        return result.overall_score

    def compute_all_metrics(
        self, question: str, context: str, answer: str
    ) -> Dict[str, Any]:
        result = self.llm_judge.evaluate(question, context, answer)
        return result.to_dict()


class EvaluationPipeline:
    def __init__(self):
        self.metrics = RagasMetrics()
        self.llm_judge = LLMJudge()

    def evaluate_query_response(
        self, question: str, context: CompressedContext, generated_answer: str
    ) -> EvaluationResult:
        logger.info(f"Evaluating response for question: '{question[:50]}...'")

        result = self.llm_judge.evaluate(
            question=question, context=context.text, answer=generated_answer
        )

        logger.info(
            f"Evaluation complete - Faithfulness: {result.faithfulness_score:.2f}, "
            f"Relevance: {result.relevance_score:.2f}, Overall: {result.overall_score:.2f}"
        )

        return result

    def evaluate_batch(
        self, test_cases: List[TestCase], query_func: callable
    ) -> Dict[str, Any]:
        results = []

        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i + 1}/{len(test_cases)}")

            try:
                context, answer = query_func(test_case.question)

                eval_result = self.evaluate_query_response(
                    question=test_case.question,
                    context=context,
                    generated_answer=answer,
                )

                results.append(
                    {
                        "test_case": test_case.question,
                        "evaluation": eval_result.to_dict(),
                    }
                )
            except Exception as e:
                logger.error(f"Test case {i + 1} failed: {e}")
                results.append(
                    {"test_case": test_case.question, "evaluation": {"error": str(e)}}
                )

        return self._aggregate_results(results)

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_results = [
            r for r in results if "evaluation" in r and "error" not in r["evaluation"]
        ]

        if not valid_results:
            return {"status": "no_valid_results", "results": results}

        avg_faithfulness = sum(
            r["evaluation"].get("faithfulness_score", 0) for r in valid_results
        ) / len(valid_results)
        avg_relevance = sum(
            r["evaluation"].get("relevance_score", 0) for r in valid_results
        ) / len(valid_results)
        avg_overall = sum(
            r["evaluation"].get("overall_score", 0) for r in valid_results
        ) / len(valid_results)

        faithful_count = sum(
            1 for r in valid_results if r["evaluation"].get("faithful", False)
        )
        relevant_count = sum(
            1 for r in valid_results if r["evaluation"].get("relevant", False)
        )

        return {
            "summary": {
                "total_tests": len(results),
                "valid_tests": len(valid_results),
                "avg_faithfulness": avg_faithfulness,
                "avg_relevance": avg_relevance,
                "avg_overall_score": avg_overall,
                "faithful_percentage": (faithful_count / len(valid_results)) * 100
                if valid_results
                else 0,
                "relevant_percentage": (relevant_count / len(valid_results)) * 100
                if valid_results
                else 0,
            },
            "grade": self._get_grade(avg_overall),
            "results": results,
        }

    def _get_grade(self, score: float) -> str:
        if score >= 0.9:
            return "A - Excellent"
        elif score >= 0.8:
            return "B - Good"
        elif score >= 0.7:
            return "C - Satisfactory"
        elif score >= 0.6:
            return "D - Needs Improvement"
        else:
            return "F - Poor"


_evaluation_pipeline_instance: Optional[EvaluationPipeline] = None


def get_evaluation_pipeline() -> EvaluationPipeline:
    global _evaluation_pipeline_instance
    if _evaluation_pipeline_instance is None:
        _evaluation_pipeline_instance = EvaluationPipeline()
    return _evaluation_pipeline_instance
