import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from app.engines.document_engine.wrapper import get_document_engine
from app.engines.csv_engine.wrapper import get_csv_engine
from retrieval.guardrails import (
    RetrievalGuardrails,
    RetrievalConfig,
    create_safe_response,
    validate_retrieval_response,
)
from core.logging import RequestTracker

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])


@router.post("/query")
async def query_dataset(
    dataset_id: str = Query(..., description="Dataset ID to query"),
    query: str = Query(..., description="Natural language query"),
):
    logger.info(f"Query request: '{query[:50]}...' on {dataset_id}")

    # If there's a JSON body, allow overriding params
    # But for now, just use query params

    return await _process_query(dataset_id, query)


async def _process_query(dataset_id: str, query: str):

    try:
        if dataset_id.startswith("ds_"):
            engine = get_csv_engine()
            result = engine.query(query, dataset_id)
            return {
                "dataset_id": dataset_id,
                "type": "csv",
                "query": query,
                "response": result,
            }
        else:
            engine = get_document_engine()
            result = engine.query(query, dataset_id)
            return {
                "dataset_id": dataset_id,
                "type": "document",
                "query": query,
                "response": result,
            }

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/{dataset_id}")
async def query_dataset_get(
    dataset_id: str,
    q: str = Query(..., alias="q"),
):
    return await query_dataset(dataset_id, q)


@router.post("/query/robust")
async def robust_query(
    dataset_id: str = Query(..., description="Dataset ID"),
    query: str = Query(..., min_length=1, max_length=2000),
    include_sources: bool = Query(True, description="Include source IDs"),
):
    """Query with retrieval guardrails and quality metrics"""
    with RequestTracker(query=query) as tracker:
        try:
            if dataset_id.startswith("ds_"):
                engine = get_csv_engine()
                result = engine.query(query, dataset_id)
                return result
            else:
                from app.engines.document_engine.wrapper import get_document_engine
                from retrieval.hybrid_retriever import get_hybrid_retriever
                from retrieval.reranker import get_reranker
                from compression.compressor import get_compressor
                from llm.reasoning import get_reasoning_pipeline
                from core.metrics import Timer

                retriever = get_hybrid_retriever()
                reranker = get_reranker()
                compressor = get_compressor()
                reasoning = get_reasoning_pipeline()

                with Timer() as t_retrieve:
                    results = retriever.retrieve(query, top_k=20)

                with Timer() as t_rerank:
                    reranked = reranker.rerank(query, results, top_k=5)
                    tracker.reranked_results_count = len(reranked)

                with Timer() as t_compress:
                    context = compressor.compress(query, reranked)

                with Timer() as t_generate:
                    query_result = reasoning.query_analysis(query, context)

                guardrails = RetrievalGuardrails(RetrievalConfig())

                retrieval_result = guardrails.apply(reranked, query)

                safe_response = create_safe_response(
                    retrieval_result,
                    query_result.answer,
                    query,
                    sources=query_result.sources if include_sources else [],
                )

                return {
                    "dataset_id": dataset_id,
                    "query": query,
                    "answer": safe_response["answer"],
                    "sources": safe_response["sources"] if include_sources else [],
                    "metadata": {
                        **safe_response["metadata"],
                        "latency_ms": {
                            "retrieval": t_retrieve.elapsed_ms,
                            "reranking": t_rerank.elapsed_ms,
                            "compression": t_compress.elapsed_ms,
                            "generation": t_generate.elapsed_ms,
                        },
                    },
                }

        except Exception as e:
            logger.error(f"Robust query error: {e}", exc_info=True)
            return {
                "error": "Query failed",
                "detail": str(e),
                "fallback_used": True,
                "answer": "I encountered an error processing your query. Please try again.",
            }
