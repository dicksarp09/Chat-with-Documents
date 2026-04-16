import logging
import re
from typing import Dict, Any, List, Optional

from app.shared.llm_clients import get_llm_client
from app.shared.schemas import AnalysisPlan, TaskItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisPlanner:
    def __init__(self):
        self.llm = get_llm_client()

    def create_plan(
        self,
        dataset_id: str,
        user_query: str,
        schema: Dict[str, Any],
        sample_rows: List[Dict[str, Any]] = None,
    ) -> AnalysisPlan:
        logger.info(f"Creating analysis plan for query: '{user_query[:50]}...'")

        schema_str = self._format_schema(schema)
        sample_str = ""
        if sample_rows:
            sample_str = f"\nSample rows:\n{self._format_samples(sample_rows)}"

        prompt = f"""You are a data analyst. Given a user query and dataset schema, create an analysis plan.

DATASET SCHEMA:
{schema_str}{sample_str}

USER QUERY:
{user_query}

TASK TYPES available:
- filter: Filter rows based on conditions
- groupby: Group data and aggregate
- correlation: Find correlations between columns
- aggregation: Calculate statistics
- sort: Sort data by columns
- transform: Apply transformations
- comparison: Compare groups or time periods
- time_series: Analyze trends over time
- top_n: Get top or bottom N records

PLANNING RULES:
1. Break the query into logical analysis tasks
2. Specify exact column names from the schema
3. Choose appropriate task types
4. Provide clear descriptions for each task

OUTPUT FORMAT (JSON only):
{{
    "tasks": [
        {{
            "type": "task_type",
            "description": "clear description of what this task does",
            "columns": ["column1", "column2"],
            "metric": "column_to_aggregate",
            "operation": "mean|sum|count|etc",
            "conditions": {{"column": "value"}} (if filter)
        }}
    ],
    "reasoning": "explain why these tasks answer the query"
}}

JSON OUTPUT:"""

        result = self.llm.generate_json(prompt)

        logger.info(f"LLM raw output: {result}")

        if not result or "tasks" not in result:
            logger.warning("Failed to create analysis plan, using fallback")
            return self._fallback_plan(dataset_id, user_query, schema)

        tasks = []
        for task_data in result.get("tasks", []):
            try:
                task = TaskItem(
                    type=task_data.get("type", "aggregation"),
                    description=task_data.get("description", ""),
                    columns=task_data.get("columns", []),
                    metric=task_data.get("metric"),
                    operation=task_data.get("operation"),
                    conditions=task_data.get("conditions"),
                )
                tasks.append(task)
            except Exception as e:
                logger.warning(f"Invalid task data: {e}")
                continue

        plan = AnalysisPlan(
            dataset_id=dataset_id,
            user_query=user_query,
            tasks=tasks,
            reasoning=result.get("reasoning", "Analysis based on query"),
        )

        logger.info(f"Created plan with {len(tasks)} tasks")
        return plan

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        lines = []

        if "columns" in schema:
            for col in schema["columns"]:
                col_type = col.get("type", "unknown")
                missing = (
                    f"{col.get('missing_pct', 0)}% missing"
                    if col.get("missing_pct", 0) > 0
                    else ""
                )
                extras = f" ({missing})" if missing else ""
                lines.append(f"  - {col['name']}: {col_type}{extras}")

        return "\n".join(lines) if lines else "No schema available"

    def _format_samples(self, sample_rows: List[Dict]) -> str:
        if not sample_rows:
            return ""

        lines = []
        for i, row in enumerate(sample_rows[:3]):
            values = [f"{k}={v}" for k, v in row.items()]
            lines.append(f"  Row {i + 1}: {', '.join(values)}")

        return "\n".join(lines)

    def _fallback_plan(
        self, dataset_id: str, user_query: str, schema: Dict[str, Any]
    ) -> AnalysisPlan:
        columns = []
        if "columns" in schema:
            columns = [col["name"] for col in schema["columns"]]

        numeric_cols = []
        categorical_cols = []

        if "columns" in schema:
            for col in schema["columns"]:
                if col.get("type") == "numeric":
                    numeric_cols.append(col["name"])
                elif col.get("type") == "categorical":
                    categorical_cols.append(col["name"])

        tasks = []

        if numeric_cols:
            tasks.append(
                TaskItem(
                    type="aggregation",
                    description="Calculate basic statistics",
                    columns=numeric_cols[:3],
                    metric=numeric_cols[0],
                    operation="describe",
                )
            )

        if categorical_cols:
            tasks.append(
                TaskItem(
                    type="groupby",
                    description=f"Group by {categorical_cols[0]}",
                    columns=[categorical_cols[0]],
                    metric=numeric_cols[0] if numeric_cols else categorical_cols[0],
                    operation="count",
                )
            )

        return AnalysisPlan(
            dataset_id=dataset_id,
            user_query=user_query,
            tasks=tasks,
            reasoning="Fallback plan - basic analysis",
        )

    def validate_plan(
        self, plan: AnalysisPlan, available_columns: List[str]
    ) -> tuple[bool, List[str]]:
        errors = []

        for task in plan.tasks:
            for col in task.columns:
                if col not in available_columns:
                    errors.append(f"Column '{col}' not in dataset")

        return len(errors) == 0, errors


def create_analysis_plan(
    dataset_id: str,
    user_query: str,
    schema: Dict[str, Any],
    sample_rows: List[Dict] = None,
) -> AnalysisPlan:
    planner = AnalysisPlanner()
    return planner.create_plan(dataset_id, user_query, schema, sample_rows)
