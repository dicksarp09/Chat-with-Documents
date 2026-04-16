import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from app.core.file_detector import detect_file_type, FileType

logger = logging.getLogger(__name__)


class EngineType(str, Enum):
    DOCUMENT = "document"
    CSV = "csv"


@dataclass
class EngineResult:
    engine_type: EngineType
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None


class UnifiedRouter:
    def __init__(self):
        self.document_engine = None
        self.csv_engine = None
        self._engines: Dict[EngineType, Any] = {}

    def register_engine(self, engine_type: EngineType, engine_instance: Any):
        self._engines[engine_type] = engine_instance
        logger.info(f"Registered engine: {engine_type.value}")

    def set_document_engine(self, engine):
        self.document_engine = engine
        self.register_engine(EngineType.DOCUMENT, engine)

    def set_csv_engine(self, engine):
        self.csv_engine = engine
        self.register_engine(EngineType.CSV, engine)

    def route_file(self, file_bytes: bytes, filename: str, **kwargs) -> EngineResult:
        file_type = detect_file_type(filename, file_bytes)

        logger.info(f"Routing file '{filename}' as {file_type.value}")

        if file_type == FileType.DOCUMENT:
            return self._process_document(file_bytes, filename, **kwargs)

        if file_type == FileType.CSV:
            return self._process_csv(file_bytes, filename, **kwargs)

        return EngineResult(
            engine_type=EngineType.DOCUMENT,
            success=False,
            data={},
            error=f"Unsupported file type: {filename}",
        )

    def _process_document(
        self, file_bytes: bytes, filename: str, **kwargs
    ) -> EngineResult:
        if EngineType.DOCUMENT not in self._engines:
            return EngineResult(
                engine_type=EngineType.DOCUMENT,
                success=False,
                data={},
                error="Document engine not available",
            )

        try:
            result = self._engines[EngineType.DOCUMENT].process(
                file_bytes, filename, **kwargs
            )
            return EngineResult(
                engine_type=EngineType.DOCUMENT,
                success=True,
                data=result,
            )
        except Exception as e:
            logger.error(f"Document engine error: {e}")
            return EngineResult(
                engine_type=EngineType.DOCUMENT,
                success=False,
                data={},
                error=str(e),
            )

    def _process_csv(self, file_bytes: bytes, filename: str, **kwargs) -> EngineResult:
        if EngineType.CSV not in self._engines:
            return EngineResult(
                engine_type=EngineType.CSV,
                success=False,
                data={},
                error="CSV engine not available",
            )

        try:
            result = self._engines[EngineType.CSV].process(
                file_bytes, filename, **kwargs
            )
            return EngineResult(
                engine_type=EngineType.CSV,
                success=True,
                data=result,
            )
        except Exception as e:
            logger.error(f"CSV engine error: {e}")
            return EngineResult(
                engine_type=EngineType.CSV,
                success=False,
                data={},
                error=str(e),
            )

    def route_query(self, query: str, dataset_id: str, **kwargs) -> EngineResult:
        dataset_type = self._get_dataset_type(dataset_id)

        if dataset_type == EngineType.DOCUMENT:
            return self._query_document(query, dataset_id, **kwargs)

        if dataset_type == EngineType.CSV:
            return self._query_csv(query, dataset_id, **kwargs)

        return EngineResult(
            engine_type=EngineType.DOCUMENT,
            success=False,
            data={},
            error=f"Unknown dataset: {dataset_id}",
        )

    def _query_document(self, query: str, dataset_id: str, **kwargs) -> EngineResult:
        if EngineType.DOCUMENT not in self._engines:
            return EngineResult(
                engine_type=EngineType.DOCUMENT,
                success=False,
                data={},
                error="Document engine not available",
            )

        try:
            result = self._engines[EngineType.DOCUMENT].query(
                query, dataset_id, **kwargs
            )
            return EngineResult(
                engine_type=EngineType.DOCUMENT,
                success=True,
                data=result,
            )
        except Exception as e:
            logger.error(f"Document query error: {e}")
            return EngineResult(
                engine_type=EngineType.DOCUMENT,
                success=False,
                data={},
                error=str(e),
            )

    def _query_csv(self, query: str, dataset_id: str, **kwargs) -> EngineResult:
        if EngineType.CSV not in self._engines:
            return EngineResult(
                engine_type=EngineType.CSV,
                success=False,
                data={},
                error="CSV engine not available",
            )

        try:
            result = self._engines[EngineType.CSV].query(query, dataset_id, **kwargs)
            return EngineResult(
                engine_type=EngineType.CSV,
                success=True,
                data=result,
            )
        except Exception as e:
            logger.error(f"CSV query error: {e}")
            return EngineResult(
                engine_type=EngineType.CSV,
                success=False,
                data={},
                error=str(e),
            )

    def _get_dataset_type(self, dataset_id: str) -> EngineType:
        if dataset_id.startswith("ds_"):
            return EngineType.CSV
        return EngineType.DOCUMENT


_router_instance: Optional[UnifiedRouter] = None


def get_router() -> UnifiedRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = UnifiedRouter()
    return _router_instance
