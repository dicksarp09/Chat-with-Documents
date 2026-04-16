import logging
import uuid
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.file_detector import detect_file_type, FileType, validate_file_size
from app.core.router import get_router, EngineType
from app.engines.document_engine.wrapper import get_document_engine
from app.engines.csv_engine.wrapper import get_csv_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Upload"])


def _get_unified_response(result):
    return {
        "dataset_id": result.data.get("dataset_id") or result.data.get("doc_id", ""),
        "type": result.engine_type.value,
        "success": result.success,
        "status": result.data.get(
            "status", "processed" if result.success else "failed"
        ),
        "result": result.data,
        "error": result.error,
    }


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"Upload request: {file.filename}")

    content = await file.read()

    is_valid, message = validate_file_size(content)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    file_type = detect_file_type(file.filename, content)

    if file_type == FileType.UNSUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.filename}. Supported: CSV, PDF, DOCX",
        )

    try:
        if file_type == FileType.CSV:
            engine = get_csv_engine()
            result = engine.process(content, file.filename)
            return {
                "dataset_id": result["dataset_id"],
                "type": "csv",
                "success": True,
                "status": result["status"],
                "result": result,
            }

        elif file_type == FileType.DOCUMENT:
            engine = get_document_engine()
            result = engine.process(content, file.filename)
            return {
                "dataset_id": result["doc_id"],
                "type": "document",
                "success": True,
                "status": result["status"],
                "result": result,
            }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def list_datasets():
    csv_engine = get_csv_engine()
    doc_engine = get_document_engine()

    datasets = []

    for ds in csv_engine.list_datasets():
        datasets.append(
            {
                "dataset_id": ds["dataset_id"],
                "type": "csv",
                "filename": ds["filename"],
                "status": ds["profile_status"],
            }
        )

    for doc in doc_engine.list_documents():
        datasets.append(
            {
                "dataset_id": doc["doc_id"],
                "type": "document",
                "filename": doc["filename"],
                "status": doc["status"],
            }
        )

    return {"datasets": datasets, "total": len(datasets)}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    if dataset_id.startswith("ds_"):
        engine = get_csv_engine()
        result = engine.get_dataset(dataset_id)
        if result:
            return result
    else:
        engine = get_document_engine()
        result = engine.get_document(dataset_id)
        if result:
            return result

    raise HTTPException(status_code=404, detail="Dataset not found")


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    if dataset_id.startswith("ds_"):
        engine = get_csv_engine()
        success = engine.delete_dataset(dataset_id)
    else:
        engine = get_document_engine()
        success = engine.delete_document(dataset_id)

    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {"status": "deleted", "dataset_id": dataset_id}


@router.get("/debug/stats")
async def get_debug_stats():
    from storage.vector_store import get_vector_store
    from retrieval.hybrid_retriever import get_hybrid_retriever

    vs = get_vector_store()
    hr = get_hybrid_retriever()

    return {
        "vector_store": vs.get_stats(),
        "hybrid_retriever": {
            "corpus_size": len(hr.corpus_ids),
            "bm25_ready": hr.bm25 is not None,
        },
    }
