import logging
from typing import List, Dict, Any, Optional
import json

from llm.groq_client import get_groq_client
from llm.prompts import (
    get_summary_prompt,
    get_key_points_prompt,
    get_risk_detection_prompt,
    get_obligation_prompt,
    get_action_prompt,
    get_query_prompt,
    get_full_analysis_prompt,
    SYSTEM_PROMPT,
)
from compression.compressor import CompressedContext
from schemas.output_schema import (
    DocumentAnalysisOutput,
    QueryOutput,
    ActionItem,
    RiskItem,
    ObligationItem,
    KeyPoint,
    Priority,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningPipeline:
    def __init__(self):
        self.llm = get_groq_client()

    def summarize(self, context: CompressedContext) -> Dict[str, Any]:
        logger.info("Running summarization...")

        prompt = get_summary_prompt(context.text)

        try:
            result = self.llm.generate_json(prompt, SYSTEM_PROMPT)

            if not result:
                result = {
                    "summary": "Unable to generate summary",
                    "document_type": None,
                    "key_theme": None,
                    "evidence": context.source_nodes[:2]
                    if context.source_nodes
                    else [],
                }

            return result

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {
                "summary": "Error during summarization",
                "document_type": None,
                "key_theme": None,
                "evidence": [],
            }

    def extract_key_points(self, context: CompressedContext) -> List[KeyPoint]:
        logger.info("Extracting key points...")

        prompt = get_key_points_prompt(context.text)

        try:
            result = self.llm.generate_json(prompt, SYSTEM_PROMPT)

            key_points = []
            for kp_data in result.get("key_points", []):
                key_points.append(
                    KeyPoint(
                        text=kp_data.get("text", ""),
                        category=kp_data.get("category", "general"),
                        evidence=kp_data.get("evidence", context.source_nodes[:2]),
                    )
                )

            return key_points

        except Exception as e:
            logger.error(f"Key points extraction error: {e}")
            return []

    def detect_risks(self, context: CompressedContext) -> List[RiskItem]:
        logger.info("Detecting risks...")

        prompt = get_risk_detection_prompt(context.text)

        try:
            result = self.llm.generate_json(prompt, SYSTEM_PROMPT)

            risks = []
            for risk_data in result.get("risks", []):
                risks.append(
                    RiskItem(
                        description=risk_data.get("description", ""),
                        severity=risk_data.get("severity", "medium"),
                        category=risk_data.get("category", "general"),
                        evidence=risk_data.get("evidence", context.source_nodes[:2]),
                    )
                )

            return risks

        except Exception as e:
            logger.error(f"Risk detection error: {e}")
            return []

    def extract_obligations(self, context: CompressedContext) -> List[ObligationItem]:
        logger.info("Extracting obligations...")

        prompt = get_obligation_prompt(context.text)

        try:
            result = self.llm.generate_json(prompt, SYSTEM_PROMPT)

            obligations = []
            for ob_data in result.get("obligations", []):
                obligations.append(
                    ObligationItem(
                        description=ob_data.get("description", ""),
                        party=ob_data.get("party", "unknown"),
                        deadline=ob_data.get("deadline"),
                        evidence=ob_data.get("evidence", context.source_nodes[:2]),
                    )
                )

            return obligations

        except Exception as e:
            logger.error(f"Obligation extraction error: {e}")
            return []

    def generate_actions(self, context: CompressedContext) -> List[ActionItem]:
        logger.info("Generating actions...")

        prompt = get_action_prompt(context.text)

        try:
            result = self.llm.generate_json(prompt, SYSTEM_PROMPT)

            actions = []
            for action_data in result.get("actions", []):
                priority_str = action_data.get("priority", "medium")
                try:
                    priority = Priority(priority_str.lower())
                except ValueError:
                    priority = Priority.MEDIUM

                actions.append(
                    ActionItem(
                        task=action_data.get("task", ""),
                        priority=priority,
                        reason=action_data.get("reason", ""),
                        evidence=action_data.get("evidence", context.source_nodes[:2]),
                    )
                )

            return actions

        except Exception as e:
            logger.error(f"Action generation error: {e}")
            return []

    def full_analysis(self, context: CompressedContext) -> DocumentAnalysisOutput:
        logger.info("Running full document analysis...")

        prompt = get_full_analysis_prompt(context.text)

        try:
            result = self.llm.generate_json(prompt, SYSTEM_PROMPT)

            key_points = []
            for kp_data in result.get("key_points", []):
                key_points.append(
                    KeyPoint(
                        text=kp_data.get("text", ""),
                        category=kp_data.get("category", "general"),
                        evidence=kp_data.get(
                            "evidence",
                            context.source_nodes[:2] if context.source_nodes else [],
                        ),
                    )
                )

            risks = []
            for risk_data in result.get("risks", []):
                risks.append(
                    RiskItem(
                        description=risk_data.get("description", ""),
                        severity=risk_data.get("severity", "medium"),
                        category=risk_data.get("category", "general"),
                        evidence=risk_data.get(
                            "evidence",
                            context.source_nodes[:2] if context.source_nodes else [],
                        ),
                    )
                )

            obligations = []
            for ob_data in result.get("obligations", []):
                obligations.append(
                    ObligationItem(
                        description=ob_data.get("description", ""),
                        party=ob_data.get("party", "unknown"),
                        deadline=ob_data.get("deadline"),
                        evidence=ob_data.get(
                            "evidence",
                            context.source_nodes[:2] if context.source_nodes else [],
                        ),
                    )
                )

            actions = []
            for action_data in result.get("actions", []):
                priority_str = action_data.get("priority", "medium")
                try:
                    priority = Priority(priority_str.lower())
                except ValueError:
                    priority = Priority.MEDIUM

                actions.append(
                    ActionItem(
                        task=action_data.get("task", ""),
                        priority=priority,
                        reason=action_data.get("reason", ""),
                        evidence=action_data.get(
                            "evidence",
                            context.source_nodes[:2] if context.source_nodes else [],
                        ),
                    )
                )

            return DocumentAnalysisOutput(
                summary=result.get("summary", ""),
                key_points=key_points,
                risks=risks,
                obligations=obligations,
                actions=actions,
                metadata={
                    "compression_ratio": context.compression_ratio,
                    "source_count": len(context.source_nodes),
                },
            )

        except Exception as e:
            logger.error(f"Full analysis error: {e}")
            return DocumentAnalysisOutput(
                summary="Error during analysis", metadata={"error": str(e)}
            )

    def query_analysis(self, query: str, context: CompressedContext) -> QueryOutput:
        logger.info(f"Running query analysis for: '{query[:50]}...'")

        prompt = get_query_prompt(query, context.text)

        try:
            result = self.llm.generate_json(prompt, SYSTEM_PROMPT)

            if not isinstance(result, dict):
                logger.error(f"LLM returned non-dict: {type(result)}")
                raise ValueError(f"Expected dict, got {type(result)}")

            evidence_nodes = result.get("evidence", [])
            if isinstance(evidence_nodes, list) and len(evidence_nodes) > 0:
                sources = evidence_nodes
            else:
                sources = context.source_nodes[:3] if context.source_nodes else []

            answer = result.get("answer", "Unable to generate answer")

            if "not found" in answer.lower() or "not in context" in answer.lower():
                confidence = "not_found"
            else:
                confidence = result.get("confidence", "explicitly_mentioned")

            key_points = []
            kp_data_list = result.get("key_points", [])
            if isinstance(kp_data_list, list):
                for kp_data in kp_data_list:
                    if isinstance(kp_data, dict):
                        key_points.append(
                            KeyPoint(
                                text=kp_data.get("text", ""),
                                category=kp_data.get("category", "general"),
                                evidence=kp_data.get("evidence", sources[:2]),
                            )
                        )

            return QueryOutput(
                answer=answer,
                summary=result.get("summary", ""),
                key_points=key_points,
                risks=[],
                obligations=[],
                actions=[],
                sources=sources,
            )

        except Exception as e:
            logger.error(f"Query analysis error: {e}")
            return QueryOutput(
                answer="Error during query analysis",
                summary="",
                sources=context.source_nodes[:3] if context.source_nodes else [],
            )


_pipeline_instance: Optional[ReasoningPipeline] = None


def get_reasoning_pipeline() -> ReasoningPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = ReasoningPipeline()
    return _pipeline_instance
