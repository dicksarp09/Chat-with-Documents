import logging
from typing import Dict, Any, List, Optional

from app.shared.llm_clients import get_llm_client
from app.shared.schemas import InsightItem, ExecutionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightGenerator:
    def __init__(self):
        self.llm = get_llm_client()

    def generate_insights(
        self, query: str, results: List[ExecutionResult], schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info(f"Generating insights for query: '{query[:50]}...'")

        context = self._build_context(results, schema)

        prompt = f"""You are a data analyst. Based on the analysis results, generate actionable insights.

USER QUERY: {query}

ANALYSIS RESULTS:
{context}

TASK:
1. Summarize the key findings (max 3)
2. Generate specific insights with supporting data
3. Provide a concise summary of the analysis
4. Suggest 2-3 follow-up questions

INSIGHT RULES:
- Only use data from the results
- Be specific with numbers and percentages
- Explain what the data means
- Highlight anomalies or notable patterns

OUTPUT FORMAT (JSON only):
{{
    "insights": [
        {{
            "text": "specific insight with data",
            "confidence": 0.95,
            "supporting_data": {{"key": "value"}}
        }}
    ],
    "summary": "concise summary paragraph",
    "follow_up_questions": ["question1", "question2"]
}}

JSON OUTPUT:"""

        response = self.llm.generate_json(prompt)

        if not response:
            return self._fallback_insights(query, results)

        insights = []
        for item in response.get("insights", []):
            insights.append(
                InsightItem(
                    text=item.get("text", ""),
                    confidence=item.get("confidence", 0.8),
                    supporting_data=item.get("supporting_data"),
                )
            )

        return {
            "insights": insights,
            "summary": response.get("summary", "Analysis complete"),
            "follow_up_questions": response.get("follow_up_questions", [])[:3],
        }

    def _build_context(
        self, results: List[ExecutionResult], schema: Dict[str, Any]
    ) -> str:
        contexts = []

        for i, result in enumerate(results):
            if not result.success:
                contexts.append(f"Analysis {i + 1}: Failed - {result.error_message}")
                continue

            data = result.data
            if data.get("type") == "dataframe":
                rows = data.get("rows", [])
                cols = data.get("columns", [])

                if rows:
                    context = f"Analysis {i + 1} (rows: {len(rows)}):\n"
                    context += f"Columns: {', '.join(cols)}\n"

                    for row in rows[:10]:
                        row_str = ", ".join([f"{k}={v}" for k, v in row.items()])
                        context += f"  - {row_str}\n"

                    contexts.append(context)

            elif data.get("type") == "scalar":
                contexts.append(f"Analysis {i + 1}: {data.get('data', 'N/A')}")

        return "\n".join(contexts) if contexts else "No results available"

    def _fallback_insights(
        self, query: str, results: List[ExecutionResult]
    ) -> Dict[str, Any]:
        insights = []

        for result in results:
            if result.success and result.data:
                data = result.data
                if data.get("type") == "dataframe":
                    rows = data.get("rows", [])
                    if rows:
                        insights.append(
                            InsightItem(
                                text=f"Found {len(rows)} records matching the criteria",
                                confidence=0.8,
                            )
                        )

        return {
            "insights": insights[:3],
            "summary": "Analysis completed successfully",
            "follow_up_questions": [
                "Would you like more details on any specific finding?",
                "Do you want to see the data broken down by another column?",
            ],
        }

    def generate_answer(
        self, query: str, insights: Dict[str, Any], schema: Dict[str, Any]
    ) -> str:
        prompt = f"""You are a data analyst. Based on the analysis insights, answer the user's question in a conversational way.

USER QUESTION: {query}

INSIGHTS:
{insights.get("summary", "No insights available")}

KEY FINDINGS:
{chr(10).join([f"- {i.text}" for i in insights.get("insights", [])[:3]])}

TASK:
Provide a clear, conversational answer that:
- Directly addresses the question
- References the key findings
- Is informative but concise
- Does not hallucinate

ANSWER:"""

        return self.llm.generate(prompt, temperature=0.3, max_tokens=500)


def generate_analysis_insights(
    query: str, results: List[ExecutionResult], schema: Dict[str, Any]
) -> Dict[str, Any]:
    generator = InsightGenerator()
    return generator.generate_insights(query, results, schema)
