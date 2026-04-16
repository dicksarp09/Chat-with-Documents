import logging
import time
import signal
from typing import Any, Dict, Optional, List
from contextlib import contextmanager
import io
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionTimeout(Exception):
    pass


class Sandbox:
    ALLOWED_LIBRARIES = {"pandas", "numpy", "math", "pd", "np"}
    FORBIDDEN_MODULES = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "openpyxl",
        "xlrd",
    }
    TIMEOUT_SECONDS = 10

    MAX_ROWS = 1_000_000
    MAX_COLUMNS = 500
    MAX_MEMORY_MB = 512
    MAX_RESULT_ROWS = 1000

    def __init__(self, timeout: int = TIMEOUT_SECONDS):
        self.timeout = timeout
        self._globals = {
            "pd": pd,
            "np": np,
            "pandas": pd,
            "numpy": np,
            "math": __import__("math"),
        }
        self._locals = {}

    def _validate_code(self, code: str) -> tuple[bool, str]:
        code_lower = code.lower()

        for module in self.FORBIDDEN_MODULES:
            if f"import {module}" in code_lower or f"from {module}" in code_lower:
                return False, f"Forbidden module: {module}"
            if f"{module}." in code_lower:
                return False, f"Forbidden module access: {module}"

        if "exec(" in code_lower or "eval(" in code_lower:
            return False, "Dynamic code execution not allowed"

        if "open(" in code_lower or "write(" in code_lower:
            return False, "File operations not allowed"

        return True, "OK"

    def validate_columns(
        self, df: pd.DataFrame, required_cols: List[str] = None
    ) -> tuple[bool, str]:
        if required_cols is None:
            return True, "OK"
        missing = set(required_cols) - set(df.columns)
        if missing:
            return False, f"Missing required columns: {missing}"
        return True, "OK"

    def validate_size(self, df: pd.DataFrame) -> tuple[bool, str]:
        row_count = len(df)
        col_count = len(df.columns)

        if row_count > self.MAX_ROWS:
            return (
                False,
                f"Data exceeds maximum rows ({self.MAX_ROWS:,}). Found {row_count:,} rows.",
            )

        if col_count > self.MAX_COLUMNS:
            return (
                False,
                f"Data exceeds maximum columns ({self.MAX_COLUMNS}). Found {col_count} columns.",
            )

        memory_bytes = df.memory_usage(deep=True).sum()
        memory_mb = memory_bytes / (1024 * 1024)

        if memory_mb > self.MAX_MEMORY_MB:
            return (
                False,
                f"Data exceeds maximum memory ({self.MAX_MEMORY_MB}MB). Found {memory_mb:.1f}MB.",
            )

        return True, "OK"

    def limit_result_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) > self.MAX_RESULT_ROWS:
            logger.info(
                f"Limiting result from {len(df)} to {self.MAX_RESULT_ROWS} rows"
            )
            return df.head(self.MAX_RESULT_ROWS)
        return df

    def execute(
        self, code: str, context: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, Any]:
        start_time = time.time()

        valid, message = self._validate_code(code)
        if not valid:
            return {
                "success": False,
                "error": message,
                "result": None,
                "execution_time_ms": 0,
            }

        if context:
            for name, df in context.items():
                self._globals[name] = df

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            result = self._execute_with_timeout(code)
            execution_time = (time.time() - start_time) * 1000

            stdout_output = sys.stdout.getvalue()
            stderr_output = sys.stderr.getvalue()

            if stderr_output:
                logger.warning(f"Sandbox stderr: {stderr_output}")

            return {
                "success": True,
                "result": result,
                "execution_time_ms": execution_time,
                "stdout": stdout_output,
                "data": self._serialize_result(result),
            }

        except ExecutionTimeout:
            return {
                "success": False,
                "error": f"Execution timeout ({self.timeout}s)",
                "result": None,
                "execution_time_ms": self.timeout * 1000,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": None,
                "execution_time_ms": (time.time() - start_time) * 1000,
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _execute_with_timeout(self, code: str) -> Any:
        compiled = compile(code, "<sandbox>", "exec")
        exec(compiled, self._globals, self._locals)
        result = self._locals.get("result")
        if result is not None:
            return result
        return self._locals

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        if result is None:
            return {"type": "none", "data": None}

        if isinstance(result, pd.DataFrame):
            return {
                "type": "dataframe",
                "shape": list(result.shape),
                "columns": list(result.columns),
                "rows": result.head(100).to_dict("records"),
                "dtypes": {col: str(dtype) for col, dtype in result.dtypes.items()},
            }

        if isinstance(result, pd.Series):
            return {
                "type": "series",
                "name": result.name,
                "data": result.head(100).tolist(),
                "index": result.head(100).index.tolist(),
            }

        if isinstance(result, dict):
            if "result" in result:
                return self._serialize_result(result["result"])
            return {"type": "dict", "data": {k: str(v) for k, v in result.items()}}

        if isinstance(result, (list, tuple)):
            return {"type": "list", "data": [str(x) for x in result[:100]]}

        if np.ndarray in type(result).__mro__:
            return {"type": "array", "data": result.tolist()[:100]}

        return {"type": "scalar", "data": str(result)}


class PandasExecutor:
    def __init__(self, sandbox: Sandbox = None):
        self.sandbox = sandbox or Sandbox()

    def execute_analysis(
        self, code: str, df: pd.DataFrame, df_name: str = "df"
    ) -> Dict[str, Any]:
        context = {df_name: df}
        result = self.sandbox.execute(code, context)
        return result

    def execute_with_validation(
        self, code: str, df: pd.DataFrame, expected_columns: List[str] = None
    ) -> Dict[str, Any]:
        if expected_columns:
            missing = set(expected_columns) - set(df.columns)
            if missing:
                return {
                    "success": False,
                    "error": f"Missing columns: {missing}",
                    "result": None,
                    "execution_time_ms": 0,
                }

        return self.execute_analysis(code, df)


_sandbox_instance: Optional[Sandbox] = None


def get_sandbox() -> Sandbox:
    global _sandbox_instance
    if _sandbox_instance is None:
        _sandbox_instance = Sandbox()
    return _sandbox_instance


def get_executor() -> PandasExecutor:
    return PandasExecutor(get_sandbox())
