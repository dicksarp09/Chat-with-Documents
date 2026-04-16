import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd

from app.shared.schemas import (
    DatasetProfile,
    QueryResponse,
    UploadResponse,
    DatasetInfo,
    ColumnProfile,
)
from app.shared.llm_clients import get_llm_client
from app.engines.csv_engine.profiler import profile_dataset
from app.engines.csv_engine.planner import create_analysis_plan
from app.engines.csv_engine.code_generator import generate_analysis_code
from app.engines.csv_engine.executor import execute_analysis
from app.engines.csv_engine.insights import generate_analysis_insights
from app.engines.csv_engine.visualization import generate_visualizations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVProcessor:
    def __init__(self):
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.llm = get_llm_client()

    def upload_csv(self, file_path: str, filename: str) -> UploadResponse:
        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"

        logger.info(f"Uploading CSV: {filename} as {dataset_id}")

        try:
            df = pd.read_csv(file_path)
            shape = list(df.shape)

            profile = profile_dataset(df, dataset_id, filename)

            self.datasets[dataset_id] = {
                "dataset_id": dataset_id,
                "filename": filename,
                "df": df,
                "profile": profile,
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
            }

            logger.info(f"CSV uploaded successfully: {dataset_id}")

            return UploadResponse(
                dataset_id=dataset_id,
                filename=filename,
                shape=shape,
                columns=list(df.columns),
                status="profiled",
                message="Dataset uploaded and profiled successfully",
            )

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return UploadResponse(
                dataset_id="",
                filename=filename,
                shape=[],
                columns=[],
                status="failed",
                message=f"Upload failed: {str(e)}",
            )

    def process_query(self, dataset_id: str, query: str) -> QueryResponse:
        start_time = time.time()

        logger.info(f"Processing query for {dataset_id}: '{query[:50]}...'")

        if dataset_id not in self.datasets:
            return QueryResponse(
                dataset_id=dataset_id,
                query=query,
                chat={
                    "answer": f"Dataset {dataset_id} not found",
                    "follow_up_questions": [],
                },
                insights=[],
                tables=[],
                plots=[],
                metadata={"error": "Dataset not found", "execution_time_ms": 0},
            )

        dataset = self.datasets[dataset_id]
        df = dataset["df"]
        profile = dataset["profile"]

        dataset["last_accessed"] = datetime.now()

        schema = self._build_schema(profile)

        plan = create_analysis_plan(dataset_id, query, schema, profile.sample_rows)

        codes = generate_analysis_code(plan, schema)

        results = execute_analysis(codes, df)

        successful_results = [r for r in results if r.success]

        if not successful_results:
            return QueryResponse(
                dataset_id=dataset_id,
                query=query,
                chat={
                    "answer": "Analysis could not be completed. Please try rephrasing your question.",
                    "follow_up_questions": [
                        "Can you provide more details?",
                        "Try asking about a specific column",
                    ],
                },
                insights=[],
                tables=[],
                plots=[],
                metadata={"execution_time_ms": (time.time() - start_time) * 1000},
            )

        insights_data = generate_analysis_insights(query, successful_results, schema)

        answer = self.llm.generate(
            f"Based on these insights: {insights_data.get('summary', '')}, "
            f"answer the question: {query}. Be concise and informative."
        )

        follow_ups = insights_data.get("follow_up_questions", [])
        if not follow_ups:
            follow_ups = [
                "Would you like to see the raw data?",
                "Do you want to analyze a different aspect?",
            ]

        tables = self._extract_tables(successful_results)

        plots = generate_visualizations(query, successful_results, schema)

        execution_time = (time.time() - start_time) * 1000

        logger.info(f"Query processed in {execution_time:.0f}ms")

        return QueryResponse(
            dataset_id=dataset_id,
            query=query,
            chat={"answer": answer, "follow_up_questions": follow_ups[:3]},
            insights=insights_data.get("insights", [])[:10],
            tables=tables[:5],
            plots=plots[:3],
            metadata={
                "execution_time_ms": round(execution_time, 2),
                "rows_processed": len(df),
                "columns_analyzed": len(plan.tasks),
            },
        )

    def _build_schema(self, profile: DatasetProfile) -> Dict[str, Any]:
        return {
            "columns": [
                {"name": col.name, "type": col.type, "missing_pct": col.missing_pct}
                for col in profile.columns
            ]
        }

    def _extract_tables(self, results: List) -> List[Dict[str, Any]]:
        tables = []

        for result in results:
            if result.success and result.data:
                data = result.data
                if data.get("type") == "dataframe":
                    tables.append(
                        {
                            "columns": data.get("columns", []),
                            "rows": data.get("rows", [])[:50],
                        }
                    )

        return tables

    def list_datasets(self) -> List[DatasetInfo]:
        datasets = []

        for ds_id, dataset in self.datasets.items():
            profile = dataset["profile"]
            datasets.append(
                DatasetInfo(
                    dataset_id=ds_id,
                    filename=dataset["filename"],
                    shape=list(profile.shape),
                    columns=[col.name for col in profile.columns],
                    profile_status="profiled",
                    created_at=dataset["created_at"],
                    memory_usage_mb=profile.memory_usage_mb,
                )
            )

        return datasets

    def get_dataset_info(self, dataset_id: str) -> Optional[DatasetInfo]:
        if dataset_id not in self.datasets:
            return None

        dataset = self.datasets[dataset_id]
        profile = dataset["profile"]

        return DatasetInfo(
            dataset_id=dataset_id,
            filename=dataset["filename"],
            shape=list(profile.shape),
            columns=[col.name for col in profile.columns],
            profile_status="profiled",
            created_at=dataset["created_at"],
            memory_usage_mb=profile.memory_usage_mb,
        )

    def delete_dataset(self, dataset_id: str) -> bool:
        if dataset_id in self.datasets:
            del self.datasets[dataset_id]
            logger.info(f"Deleted dataset: {dataset_id}")
            return True
        return False


_processor_instance: Optional[CSVProcessor] = None


def get_processor() -> CSVProcessor:
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = CSVProcessor()
    return _processor_instance
