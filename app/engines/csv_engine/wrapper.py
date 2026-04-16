import logging
import uuid
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from app.engines.csv_engine.processor import get_processor as get_csv_processor
from app.engines.csv_engine.profiler import profile_dataset
from app.core.file_detector import FileType, detect_file_type

logger = logging.getLogger(__name__)


class CSVEngineWrapper:
    def __init__(self):
        self.processor = None
        self._initialized = False

    def _ensure_init(self):
        if not self._initialized:
            self.processor = get_csv_processor()
            self._initialized = True

    def process(self, file_bytes: bytes, filename: str, **kwargs) -> Dict[str, Any]:
        self._ensure_init()

        file_type = detect_file_type(filename, file_bytes)
        if file_type != FileType.CSV:
            raise ValueError(f"Not a CSV file: {filename}")

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            result = self.processor.upload_csv(tmp_path, filename)

            return {
                "dataset_id": result.dataset_id,
                "filename": result.filename,
                "type": "csv",
                "shape": result.shape,
                "columns": result.columns,
                "status": result.status,
                "message": result.message,
            }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def query(self, query: str, dataset_id: str, **kwargs) -> Dict[str, Any]:
        self._ensure_init()
        result = self.processor.process_query(dataset_id, query)
        return result.model_dump()

    def list_datasets(self) -> list:
        self._ensure_init()
        datasets = self.processor.list_datasets()
        return [d.model_dump() for d in datasets]

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_init()
        result = self.processor.get_dataset_info(dataset_id)
        return result.model_dump() if result else None

    def delete_dataset(self, dataset_id: str) -> bool:
        self._ensure_init()
        return self.processor.delete_dataset(dataset_id)


_csv_engine_instance: Optional[CSVEngineWrapper] = None


def get_csv_engine() -> CSVEngineWrapper:
    global _csv_engine_instance
    if _csv_engine_instance is None:
        _csv_engine_instance = CSVEngineWrapper()
    return _csv_engine_instance
