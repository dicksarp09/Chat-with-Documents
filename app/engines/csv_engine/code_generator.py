import logging
from typing import Dict, Any, List, Optional

from app.shared.llm_clients import get_llm_client
from app.shared.schemas import AnalysisPlan, TaskItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeGenerator:
    def __init__(self):
        self.llm = get_llm_client()

    def generate_code(
        self, plan: AnalysisPlan, schema: Dict[str, Any], df_name: str = "df"
    ) -> List[str]:
        logger.info(f"Generating code for {len(plan.tasks)} tasks")

        codes = []

        for i, task in enumerate(plan.tasks):
            code = self._generate_task_code(task, schema, df_name, i)
            if code:
                codes.append(code)

        if not codes:
            codes.append(self._generate_fallback_code(schema, df_name))

        logger.info(f"Generated {len(codes)} code blocks")
        return codes

    def _generate_task_code(
        self, task: TaskItem, schema: Dict[str, Any], df_name: str, task_index: int
    ) -> Optional[str]:
        valid_columns = self._get_valid_columns(task.columns, schema)
        all_columns = self._get_all_columns(schema)

        if not valid_columns:
            logger.warning(f"No valid columns for task: {task.type}")
            return None

        if task.type == "filter":
            return self._generate_filter_code(task, df_name, valid_columns)

        elif task.type == "groupby":
            return self._generate_groupby_code(
                task, df_name, valid_columns, all_columns
            )

        elif task.type == "aggregation":
            return self._generate_aggregation_code(task, df_name, valid_columns)

        elif task.type == "correlation":
            return self._generate_correlation_code(task, df_name, valid_columns)

        elif task.type == "sort":
            return self._generate_sort_code(task, df_name, valid_columns)

        elif task.type == "top_n":
            return self._generate_top_n_code(task, df_name, valid_columns)

        elif task.type == "time_series":
            return self._generate_time_series_code(task, df_name, valid_columns)

        elif task.type == "comparison":
            return self._generate_comparison_code(task, df_name, valid_columns)

        else:
            return self._generate_generic_code(task, df_name, valid_columns)

    def _get_valid_columns(
        self, columns: List[str], schema: Dict[str, Any]
    ) -> List[str]:
        available = set()
        if "columns" in schema:
            available = {col["name"] for col in schema["columns"]}

        return [col for col in columns if col in available]

    def _get_all_columns(self, schema: Dict[str, Any]) -> List[str]:
        if "columns" in schema:
            return [col["name"] for col in schema["columns"]]
        return []

    def _generate_filter_code(
        self, task: TaskItem, df_name: str, columns: List[str]
    ) -> str:
        col = columns[0]
        conditions = task.conditions or {}

        filter_expr = ""
        if conditions:
            for k, v in conditions.items():
                if k in columns:
                    filter_expr = f'{df_name}["{k}"] == {repr(v)}'
        elif task.description:
            filter_expr = f'{df_name}["{col}"].notna()'

        if not filter_expr:
            filter_expr = f'{df_name}["{col}"].notna()'

        return f"# Task: {task.description}\nresult = {df_name}[{filter_expr}].copy()\nprint(result.shape)"

    def _generate_groupby_code(
        self,
        task: TaskItem,
        df_name: str,
        columns: List[str],
        all_columns: List[str] = None,
    ) -> str:
        group_col = columns[0]
        search_columns = all_columns if all_columns else columns

        # Known numeric column names
        known_numeric = {
            "sales",
            "quantity",
            "price",
            "amount",
            "total",
            "revenue",
            "cost",
            "count",
            "value",
            "average",
            "mean",
            "min",
            "max",
            "profit",
            "discount",
            "tax",
        }

        # Find metric column - prefer task.metric, then look in all columns
        metric_str = str(task.metric) if task.metric else None
        metric_col = None

        if metric_str:
            # Clean up the metric string
            cleaned_metric = (
                metric_str.replace('"', "")
                .replace("'", "")
                .replace("[", "")
                .replace("]", "")
                .strip()
            )
            # Check if it's in search_columns first
            if cleaned_metric in search_columns:
                metric_col = cleaned_metric
            # Otherwise, check if any column name matches (case insensitive)
            elif cleaned_metric.lower() in [c.lower() for c in search_columns]:
                metric_col = next(
                    c for c in search_columns if c.lower() == cleaned_metric.lower()
                )

        # If still no metric found, look for numeric columns
        if not metric_col:
            # Try to find a numeric column in the known list
            for col in search_columns:
                if col.lower() in known_numeric:
                    metric_col = col
                    break
            # If still nothing, just use the first column that might be numeric based on name
            if not metric_col:
                for col in search_columns:
                    if any(n in col.lower() for n in known_numeric):
                        metric_col = col
                        break

        # Last resort - use group_col (will produce incorrect results but won't crash)
        if not metric_col:
            metric_col = group_col

        operation = task.operation or "count"
        if isinstance(operation, list):
            operation = str(operation[0]) if operation else "count"
        operation = str(operation).lower().strip()

        agg_map = {
            "count": "size()",
            "sum": f'["{metric_col}"].sum()',
            "mean": f'["{metric_col}"].mean()',
            "avg": f'["{metric_col}"].mean()',
            "min": f'["{metric_col}"].min()',
            "max": f'["{metric_col}"].max()',
        }

        agg_func = agg_map.get(operation, "size()")

        logger.info(
            f"Groupby: group_col={group_col}, metric_col={metric_col}, operation={operation}"
        )

        # Special handling for size() which doesn't need column reference
        if operation == "count":
            return f'# Task: {task.description}\nresult = {df_name}.groupby("{group_col}").{agg_func}.reset_index()\nprint(result)'
        else:
            return f'# Task: {task.description}\nresult = {df_name}.groupby("{group_col}"){agg_func}.reset_index()\nprint(result)'

    def _generate_aggregation_code(
        self, task: TaskItem, df_name: str, columns: List[str]
    ) -> str:
        metric_col = (
            task.metric if task.metric and str(task.metric) in columns else columns[0]
        )
        operation = task.operation or "describe"

        if isinstance(operation, list):
            operation = str(operation[0]) if operation else "describe"
        operation = str(operation).lower().strip()

        if metric_col not in columns:
            metric_col = columns[0]

        if operation == "describe":
            col_list = ", ".join([f'"{c}"' for c in columns])
            return f"# Task: {task.description}\nresult = {df_name}[[{col_list}]].describe()\nprint(result)"

        valid_ops = ["sum", "mean", "avg", "min", "max", "count", "std", "var"]
        if operation not in valid_ops:
            operation = "describe"

        if operation == "describe":
            col_list = ", ".join([f'"{c}"' for c in columns])
            return f"# Task: {task.description}\nresult = {df_name}[[{col_list}]].describe()\nprint(result)"

        return f'# Task: {task.description}\nresult = {df_name}["{metric_col}"].{operation}()\nprint(result)'

    def _generate_correlation_code(
        self, task: TaskItem, df_name: str, columns: List[str]
    ) -> str:
        cols = columns[:5] if len(columns) > 5 else columns
        cols_str = ", ".join([f'"{c}"' for c in cols])

        return f"# Task: {task.description}\nresult = {df_name}[[{cols_str}]].corr()\nprint(result)"

    def _generate_sort_code(
        self, task: TaskItem, df_name: str, columns: List[str]
    ) -> str:
        sort_col = (
            task.metric if task.metric and str(task.metric) in columns else columns[0]
        )
        ascending = "True" if "asc" in task.description.lower() else "False"

        return f'# Task: {task.description}\nresult = {df_name}.sort_values("{sort_col}", ascending={ascending}).head(20)\nprint(result)'

    def _generate_top_n_code(
        self, task: TaskItem, df_name: str, columns: List[str]
    ) -> str:
        sort_col = task.metric or columns[0]
        n = 10

        if "bottom" in task.description.lower():
            return f'# Task: {task.description}\nresult = {df_name}.nsmallest({n}, "{sort_col}")\nprint(result)'

        return f'# Task: {task.description}\nresult = {df_name}.nlargest({n}, "{sort_col}")\nprint(result)'

    def _generate_time_series_code(
        self, task: TaskItem, df_name: str, columns: List[str]
    ) -> str:
        date_col = columns[0]
        metric_col = task.metric or (columns[1] if len(columns) > 1 else columns[0])

        return f"# Task: {task.description}\nresult = {df_name}.groupby(pd.to_datetime({df_name}[\"{date_col}\"], errors='coerce').dt.to_period('M'))[\"{metric_col}\"].mean().reset_index()\nprint(result)"

    def _generate_comparison_code(
        self, task: TaskItem, df_name: str, columns: List[str]
    ) -> str:
        col = columns[0]

        return f"# Task: {task.description}\nresult = {df_name}.groupby({df_name}[\"{col}\"].astype(str)).agg(['mean', 'count', 'std']).reset_index()\nprint(result)"

    def _generate_generic_code(
        self, task: TaskItem, df_name: str, columns: List[str]
    ) -> str:
        cols_str = ", ".join([f'"{c}"' for c in columns[:5]])

        return f"# Task: {task.description}\nresult = {df_name}[[{cols_str}]].head(20)\nprint(result)"

    def _generate_fallback_code(self, schema: Dict[str, Any], df_name: str) -> str:
        return f"# Default analysis\nresult = {df_name}.head(20)\nprint(result)"


def generate_analysis_code(plan: AnalysisPlan, schema: Dict[str, Any]) -> List[str]:
    generator = CodeGenerator()
    return generator.generate_code(plan, schema)
