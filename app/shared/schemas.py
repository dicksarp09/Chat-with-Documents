from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    type: Literal["numeric", "categorical", "datetime", "boolean", "text", "unknown"]
    count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    sample_values: List[Any] = Field(default_factory=list)

    numeric_stats: Optional[Dict[str, float]] = None
    categorical_stats: Optional[Dict[str, Any]] = None


class DatasetProfile(BaseModel):
    dataset_id: str
    filename: str
    shape: tuple[int, int]
    columns: List[ColumnProfile]
    memory_usage_mb: float
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)


class TaskItem(BaseModel):
    type: Literal[
        "filter",
        "groupby",
        "correlation",
        "aggregation",
        "sort",
        "transform",
        "comparison",
        "time_series",
        "top_n",
    ]
    description: str
    columns: List[str] = Field(default_factory=list)
    metric: Optional[str] = None
    operation: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None


class AnalysisPlan(BaseModel):
    dataset_id: str
    user_query: str
    tasks: List[TaskItem]
    reasoning: str


class ExecutionResult(BaseModel):
    success: bool
    result_type: Literal["table", "dataframe", "series", "scalar", "error"]
    data: Any
    row_count: int = 0
    column_count: int = 0
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None


class PlotSpec(BaseModel):
    type: Literal["bar", "line", "scatter", "pie", "histogram", "box", "heatmap"]
    x: str
    y: str
    title: str = ""
    color_by: Optional[str] = None
    aggregation: Optional[str] = None


class InsightItem(BaseModel):
    text: str
    confidence: float = 1.0
    supporting_data: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    dataset_id: str
    query: str
    chat: Dict[str, Any]
    insights: List[InsightItem]
    tables: List[Dict[str, Any]]
    plots: List[PlotSpec]
    metadata: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "ds_123",
                "query": "Why is revenue dropping?",
                "chat": {
                    "answer": "Revenue dropped 15% due to seasonal factors...",
                    "follow_up_questions": ["Do you want country breakdown?"],
                },
                "insights": [
                    {"text": "Q2 had lowest revenue in 3 years", "confidence": 0.95}
                ],
                "tables": [
                    {
                        "columns": ["month", "revenue"],
                        "rows": [["Jan", 1000], ["Feb", 950]],
                    }
                ],
                "plots": [
                    {
                        "type": "line",
                        "x": "month",
                        "y": "revenue",
                        "title": "Revenue Trend",
                    }
                ],
                "metadata": {"execution_time_ms": 850, "rows_processed": 5000},
            }
        }


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    shape: List[int]
    columns: List[str]
    status: str
    message: str


class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    shape: List[int]
    columns: List[str]
    profile_status: str
    created_at: datetime
    memory_usage_mb: float
