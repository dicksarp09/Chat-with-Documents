import logging
import uuid
import os
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import aiofiles

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from schemas.output_schema import (
    UploadResponse,
    QueryRequest,
    QueryOutput,
    AnalyzeRequest,
    DocumentAnalysisOutput,
    DocumentInfo,
    ParsedDocument,
    DocumentStructure,
)
from parsers.docx_parser import parse_docx
from parsers.pdf_parser import parse_pdf
from chunking.hierarchical_chunker import chunk_document
from embeddings.embedder import get_embedder
from storage.vector_store import get_vector_store, VectorStore
from retrieval.hybrid_retriever import get_hybrid_retriever
from retrieval.reranker import get_reranker
from compression.compressor import get_compressor
from llm.reasoning import get_reasoning_pipeline
from validation.json_validator import get_validator, ValidationResult
from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

DOCUMENT_STORAGE = Path(
    "C:/Users/Dickson/Desktop/Document Intelligence folder/storage/documents"
)
DOCUMENT_STORAGE.mkdir(parents=True, exist_ok=True)

document_registry: Dict[str, Dict[str, Any]] = {}


def get_vector_store_instance() -> VectorStore:
    return get_vector_store()


def get_hybrid_retriever_instance():
    return get_hybrid_retriever()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    logger.info(f"Received upload request for file: {file.filename}")

    try:
        file_id = str(uuid.uuid4())
        file_path = DOCUMENT_STORAGE / f"{file_id}_{file.filename}"

        content = await file.read()

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        logger.info(f"Saved file to {file_path}")

        background_tasks.add_task(process_document, file_id, file_path, file.filename)

        return UploadResponse(
            doc_id=file_id,
            filename=file.filename,
            status="processing",
            chunk_count=0,
            message="Document uploaded successfully and is being processed",
        )

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def process_document(doc_id: str, file_path: Path, filename: str):
    logger.info(f"Processing document: {doc_id}")

    try:
        if filename.lower().endswith(".docx"):
            parsed_doc = parse_docx(str(file_path))
        elif filename.lower().endswith(".pdf"):
            parsed_doc = parse_pdf(str(file_path))
        else:
            logger.error(f"Unsupported file type: {filename}")
            return

        logger.info(f"Parsed document: {len(parsed_doc.sections)} sections")

        document_structure = chunk_document(parsed_doc)
        logger.info(f"Created {len(document_structure.nodes)} chunks")

        vector_store = get_vector_store_instance()
        vector_store.add_nodes(document_structure.nodes)

        hybrid_retriever = get_hybrid_retriever_instance()
        hybrid_retriever.rebuild_index(doc_id)

        document_registry[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "file_path": str(file_path),
            "file_type": parsed_doc.file_type,
            "uploaded_at": datetime.now(),
            "chunk_count": document_structure.total_chunks,
            "status": "processed",
            "parsed_doc": parsed_doc,
            "structure": document_structure,
        }

        logger.info(f"Document {doc_id} processing completed")

    except Exception as e:
        logger.error(f"Processing error for {doc_id}: {e}")
        if doc_id in document_registry:
            document_registry[doc_id]["status"] = "failed"
            document_registry[doc_id]["error"] = str(e)


@router.post("/query", response_model=QueryOutput)
async def query_documents(request: QueryRequest):
    logger.info(f"Query request: '{request.query[:50]}...' (doc_id: {request.doc_id})")

    try:
        hybrid_retriever = get_hybrid_retriever_instance()
        reranker = get_reranker()
        compressor = get_compressor()
        reasoning = get_reasoning_pipeline()
        validator = get_validator()

        retrieved = hybrid_retriever.retrieve(
            request.query, doc_id=request.doc_id, top_k=20
        )
        logger.info(f"Retrieved {len(retrieved)} results from hybrid search")

        if not retrieved:
            return QueryOutput(
                answer="No relevant documents found.",
                summary="",
                key_points=[],
                risks=[],
                obligations=[],
                actions=[],
                sources=[],
            )

        if settings.use_reranker:
            reranked = reranker.rerank(request.query, retrieved, top_k=5)
            logger.info(f"Reranked to top {len(reranked)} results")
        else:
            reranked = retrieved[:5]
            logger.info(f"Skipped reranker (disabled), using top {len(reranked)} results")

        compressed = compressor.compress(request.query, reranked)
        logger.info(f"Compressed context: {compressed.compression_ratio:.2%}")

        query_result = reasoning.query_analysis(request.query, compressed)

        validation = validator.validate_with_fallback(
            query_result.model_dump(), "query"
        )

        if not validation.is_valid:
            fixed = validator.fix_common_errors(validation.raw_data, "query")
            validation = validator.validate_query_output(fixed)

        if validation.is_valid:
            return validation.data
        else:
            return QueryOutput(
                answer=query_result.answer,
                summary=query_result.summary,
                key_points=query_result.key_points,
                risks=query_result.risks,
                obligations=query_result.obligations,
                actions=query_result.actions,
                sources=query_result.sources,
            )

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=DocumentAnalysisOutput)
async def analyze_document(request: AnalyzeRequest):
    logger.info(f"Analysis request (doc_id: {request.doc_id})")

    try:
        if not request.doc_id:
            vector_store = get_vector_store_instance()
            all_nodes = vector_store.get_all_nodes()

            if not all_nodes:
                raise HTTPException(status_code=404, detail="No documents found")

            compressor = get_compressor()
            reasoning = get_reasoning_pipeline()
            validator = get_validator()

            from compression.compressor import CompressedContext

            context = CompressedContext(
                text=" ".join([n.text for n in all_nodes[:50]]),
                source_nodes=[n.id for n in all_nodes[:50]],
                compression_ratio=1.0,
                original_length=sum(len(n.text) for n in all_nodes[:50]),
                compressed_length=sum(len(n.text) for n in all_nodes[:50]),
            )

            analysis = reasoning.full_analysis(context)

            validation = validator.validate_with_fallback(
                analysis.model_dump(), "document_analysis"
            )

            if validation.is_valid:
                return validation.data

            return analysis

        if request.doc_id not in document_registry:
            raise HTTPException(status_code=404, detail="Document not found")

        doc_data = document_registry[request.doc_id]
        structure = doc_data["structure"]

        compressor = get_compressor()
        reasoning = get_reasoning_pipeline()
        validator = get_validator()

        context_texts = [n.text for n in structure.nodes[:100]]
        context = CompressedContext(
            text=" ".join(context_texts),
            source_nodes=[n.id for n in structure.nodes[:100]],
            compression_ratio=1.0,
            original_length=sum(len(t) for t in context_texts),
            compressed_length=sum(len(t) for t in context_texts),
        )

        analysis = reasoning.full_analysis(context)

        validation = validator.validate_with_fallback(
            analysis.model_dump(), "document_analysis"
        )

        if not validation.is_valid:
            fixed = validator.fix_common_errors(
                validation.raw_data, "document_analysis"
            )
            validation = validator.validate_document_analysis(fixed)

        if validation.is_valid:
            return validation.data

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str):
    logger.info(f"Get document request: {doc_id}")

    if doc_id not in document_registry:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_data = document_registry[doc_id]

    return DocumentInfo(
        doc_id=doc_id,
        filename=doc_data["filename"],
        file_type=doc_data["file_type"],
        uploaded_at=doc_data["uploaded_at"],
        chunk_count=doc_data["chunk_count"],
        status=doc_data["status"],
    )


@router.get("/documents")
async def list_documents():
    logger.info("List documents request")

    docs = []
    for doc_id, doc_data in document_registry.items():
        docs.append(
            DocumentInfo(
                doc_id=doc_id,
                filename=doc_data["filename"],
                file_type=doc_data["file_type"],
                uploaded_at=doc_data["uploaded_at"],
                chunk_count=doc_data["chunk_count"],
                status=doc_data["status"],
            )
        )

    return {"documents": docs, "total": len(docs)}


@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    logger.info(f"Delete document request: {doc_id}")

    if doc_id not in document_registry:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_data = document_registry[doc_id]

    file_path = Path(doc_data["file_path"])
    if file_path.exists():
        file_path.unlink()

    vector_store = get_vector_store_instance()
    vector_store.delete_document(doc_id)

    hybrid_retriever = get_hybrid_retriever_instance()
    hybrid_retriever.rebuild_index()

    del document_registry[doc_id]

    return {"status": "deleted", "doc_id": doc_id}


@router.get("/stats")
async def get_stats():
    logger.info("Stats request")

    vector_store = get_vector_store_instance()
    store_stats = vector_store.get_stats()

    hybrid_retriever = get_hybrid_retriever_instance()
    retriever_stats = hybrid_retriever.get_retrieval_stats()

    return {
        "vector_store": store_stats,
        "retriever": retriever_stats,
        "registered_documents": len(document_registry),
    }


# CSV Intelligence Engine Routes
csv_router = APIRouter(prefix="/csv", tags=["CSV Intelligence"])

CSV_DATASET_STORAGE: Dict[str, Dict[str, Any]] = {}


@csv_router.post("/upload", response_model=dict)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        import tempfile
        import pandas as pd
        from app.engines.csv_engine.profiler import profile_dataset
        from app.engines.csv_engine.processor import get_processor

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        processor = get_processor()
        result = processor.upload_csv(tmp_path, file.filename)

        os.unlink(tmp_path)

        return {
            "dataset_id": result.dataset_id,
            "filename": result.filename,
            "shape": result.shape,
            "columns": result.columns,
            "status": result.status,
            "message": result.message,
        }

    except Exception as e:
        logger.error(f"CSV upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@csv_router.post("/query", response_model=dict)
async def query_csv(dataset_id: str, query: str):
    from app.engines.csv_engine.processor import get_processor

    processor = get_processor()
    result = processor.process_query(dataset_id, query)
    return result.model_dump()


@csv_router.get("/datasets", response_model=list)
async def list_csv_datasets():
    from app.engines.csv_engine.processor import get_processor

    processor = get_processor()
    datasets = processor.list_datasets()
    return [d.model_dump() for d in datasets]


@csv_router.get("/datasets/{dataset_id}", response_model=dict)
async def get_csv_dataset(dataset_id: str):
    from app.engines.csv_engine.processor import get_processor

    processor = get_processor()
    result = processor.get_dataset_info(dataset_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return result.model_dump()


@csv_router.delete("/datasets/{dataset_id}")
async def delete_csv_dataset(dataset_id: str):
    from app.engines.csv_engine.processor import get_processor

    processor = get_processor()
    success = processor.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"status": "deleted", "dataset_id": dataset_id}
