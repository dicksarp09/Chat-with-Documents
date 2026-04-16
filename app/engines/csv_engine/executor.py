import logging
import time
from typing import Dict, Any, List, Optional

import pandas as pd

from app.shared.sandbox import get_sandbox, Sandbox
from app.shared.schemas import ExecutionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, sandbox: Sandbox = None):
        self.sandbox = sandbox or get_sandbox()

    def execute(
        self,
        code: str,
        df: pd.DataFrame,
        df_name: str = "df",
        expected_columns: List[str] = None,
    ) -> ExecutionResult:
        start_time = time.time()

        valid_size, size_msg = self.sandbox.validate_size(df)
        if not valid_size:
            logger.error(f"Data size validation failed: {size_msg}")
            return ExecutionResult(
                success=False,
                result_type="error",
                data=None,
                error_message=size_msg,
                execution_time_ms=0,
                row_count=0,
                column_count=0,
            )

        if expected_columns:
            valid_cols, cols_msg = self.sandbox.validate_columns(df, expected_columns)
            if not valid_cols:
                logger.error(f"Column validation failed: {cols_msg}")
                return ExecutionResult(
                    success=False,
                    result_type="error",
                    data=None,
                    error_message=cols_msg,
                    execution_time_ms=0,
                    row_count=len(df),
                    column_count=len(df.columns) if hasattr(df, "columns") else 0,
                )

        logger.info(f"Executing code: {code[:100]}...")

        result = self.sandbox.execute(code, {df_name: df})

        execution_time = (time.time() - start_time) * 1000

        if not result["success"]:
            return ExecutionResult(
                success=False,
                result_type="error",
                data=None,
                error_message=result.get("error", "Unknown error"),
                execution_time_ms=execution_time,
            )

        serialized = result.get("data", {})
        result_type = serialized.get("type", "unknown")

        return ExecutionResult(
            success=True,
            result_type=result_type,
            data=serialized,
            row_count=serialized.get("shape", [0, 0])[0]
            if result_type == "dataframe"
            else 0,
            column_count=serialized.get("shape", [0, 0])[1]
            if result_type == "dataframe"
            else 0,
            execution_time_ms=execution_time,
        )

    def execute_batch(
        self, codes: List[str], df: pd.DataFrame, df_name: str = "df"
    ) -> List[ExecutionResult]:
        results = []

        for code in codes:
            result = self.execute(code, df, df_name)
            results.append(result)

            if not result.success:
                logger.warning(f"Execution failed: {result.error_message}")

        return results

    def validate_result(self, result: ExecutionResult) -> tuple[bool, str]:
        if not result.success:
            return False, result.error_message or "Execution failed"

        if result.result_type == "error":
            return False, "Result type is error"

        if result.data is None:
            return False, "No data returned"

        return True, "OK"


def execute_analysis(
    codes: List[str], df: pd.DataFrame, df_name: str = "df"
) -> List[ExecutionResult]:
    executor = Executor()
    return executor.execute_batch(codes, df, df_name)
