import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from app.shared.schemas import DatasetProfile, ColumnProfile
from app.shared.sandbox import get_executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVProfiler:
    def __init__(self):
        self.executor = get_executor()

    def profile(
        self, df: pd.DataFrame, dataset_id: str, filename: str
    ) -> DatasetProfile:
        logger.info(f"Profiling dataset {dataset_id}: {df.shape}")

        columns = []
        for col in df.columns:
            col_profile = self._profile_column(df[col], col)
            columns.append(col_profile)

        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        sample_rows = df.head(5).to_dict("records")

        profile = DatasetProfile(
            dataset_id=dataset_id,
            filename=filename,
            shape=df.shape,
            columns=columns,
            memory_usage_mb=round(memory_mb, 2),
            sample_rows=sample_rows,
        )

        logger.info(
            f"Profile complete for {dataset_id}: {len(columns)} columns profiled"
        )
        return profile

    def _profile_column(self, series: pd.Series, name: str) -> ColumnProfile:
        dtype = str(series.dtype)
        count = len(series)
        missing_count = series.isna().sum()
        missing_pct = round(missing_count / count * 100, 2) if count > 0 else 0
        unique_count = series.nunique(dropna=False)
        unique_pct = round(unique_count / count * 100, 2) if count > 0 else 0

        sample_values = series.dropna().head(5).tolist()

        col_type = self._detect_type(series, dtype)

        numeric_stats = None
        categorical_stats = None

        if col_type == "numeric":
            numeric_stats = self._get_numeric_stats(series)
        elif col_type == "categorical":
            categorical_stats = self._get_categorical_stats(series)

        return ColumnProfile(
            name=name,
            dtype=dtype,
            type=col_type,
            count=count,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            sample_values=sample_values,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
        )

    def _detect_type(self, series: pd.Series, dtype: str) -> str:
        dtype_lower = dtype.lower()

        if "int" in dtype_lower or "float" in dtype_lower:
            if series.nunique() <= 2:
                return "boolean"
            return "numeric"

        if "datetime" in dtype_lower or "date" in dtype_lower:
            return "datetime"

        if "object" in dtype_lower or "str" in dtype_lower:
            if series.nunique() < 50:
                return "categorical"
            sample_str = (
                str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else ""
            )
            if len(sample_str) > 100:
                return "text"
            return "categorical"

        if "bool" in dtype_lower:
            return "boolean"

        return "unknown"

    def _get_numeric_stats(self, series: pd.Series) -> Dict[str, float]:
        stats = series.describe()
        return {
            "mean": round(float(stats.get("mean", 0) or 0), 2),
            "std": round(float(stats.get("std", 0) or 0), 2),
            "min": round(float(stats.get("min", 0) or 0), 2),
            "max": round(float(stats.get("max", 0) or 0), 2),
            "q25": round(float(stats.get("25%", 0) or 0), 2),
            "q50": round(float(stats.get("50%", 0) or 0), 2),
            "q75": round(float(stats.get("75%", 0) or 0), 2),
        }

    def _get_categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        value_counts = series.value_counts().head(10)
        return {
            "top_values": {str(k): int(v) for k, v in value_counts.items()},
            "mode": str(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
        }

    def get_schema_summary(self, profile: DatasetProfile) -> str:
        lines = [
            f"Dataset: {profile.filename}",
            f"Shape: {profile.shape[0]} rows x {profile.shape[1]} columns",
            f"Memory: {profile.memory_usage_mb:.2f} MB",
            "",
            "Columns:",
        ]

        for col in profile.columns:
            missing = f"{col.missing_pct}%" if col.missing_pct > 0 else ""
            unique = f"{col.unique_count} unique" if col.unique_count < 50 else ""

            extras = ", ".join(filter(None, [missing, unique]))
            if extras:
                extras = f" ({extras})"

            lines.append(f"  - {col.name}: {col.type}{extras}")

        return "\n".join(lines)


def profile_dataset(df: pd.DataFrame, dataset_id: str, filename: str) -> DatasetProfile:
    profiler = CSVProfiler()
    return profiler.profile(df, dataset_id, filename)
