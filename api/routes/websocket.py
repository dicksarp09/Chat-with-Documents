import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect, Query

from app.engines.csv_engine.wrapper import CSVEngineWrapper
from app.engines.document_engine.wrapper import DocumentEngineWrapper
from storage.vector_store import get_vector_store
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

vector_store = None
csv_engine = None
document_engine = None
active_connections: dict[str, list[WebSocket]] = {}


def get_vector_store_instance():
    global vector_store
    if vector_store is None:
        vector_store = get_vector_store()
    return vector_store


def get_csv_engine():
    global csv_engine
    if csv_engine is None:
        csv_engine = CSVEngineWrapper()
    return csv_engine


def get_document_engine():
    global document_engine
    if document_engine is None:
        document_engine = DocumentEngineWrapper(get_vector_store_instance())
    return document_engine


async def send_json(websocket: WebSocket, data: dict[str, Any]):
    await websocket.send_text(json.dumps(data))


async def send_chunk(websocket: WebSocket, content: str):
    await send_json(
        websocket,
        {
            "type": "chunk",
            "content": content,
            "final": False,
        },
    )


async def send_complete(websocket: WebSocket, final_data: dict[str, Any]):
    await send_json(
        websocket,
        {
            "type": "complete",
            "final": True,
            "data": final_data,
        },
    )


async def send_error(websocket: WebSocket, error: str):
    await send_json(
        websocket,
        {
            "type": "error",
            "error": error,
        },
    )


async def process_csv_query(
    websocket: WebSocket,
    dataset_id: str,
    query: str,
):
    engine = get_csv_engine()

    try:
        dataset = engine.get_dataset_info(dataset_id)
        if not dataset:
            await send_error(websocket, f"Dataset {dataset_id} not found")
            return

        answer = ""
        async for chunk in engine.query_streaming(dataset_id, query):
            answer += chunk
            await send_chunk(websocket, chunk)
            await asyncio.sleep(0.01)

        result = engine.get_result(dataset_id)

        await send_complete(
            websocket,
            {
                "type": "csv",
                "response": {
                    "answer": answer,
                    "tables": result.get("tables", []) if result else [],
                    "plots": result.get("plots", []) if result else [],
                    "insights": result.get("insights", []) if result else [],
                },
            },
        )

    except Exception as e:
        logger.error(f"CSV query error: {e}")
        await send_error(websocket, str(e))


async def process_document_query(
    websocket: WebSocket,
    dataset_id: str,
    query: str,
):
    engine = get_document_engine()

    try:
        dataset = engine.get_dataset_info(dataset_id)
        if not dataset:
            await send_error(websocket, f"Dataset {dataset_id} not found")
            return

        retriever = HybridRetriever(get_vector_store_instance())

        answer = ""
        async for chunk in engine.query_streaming(dataset_id, query, retriever):
            answer += chunk
            await send_chunk(websocket, chunk)
            await asyncio.sleep(0.01)

        result = engine.get_result(dataset_id)

        await send_complete(
            websocket,
            {
                "type": "document",
                "response": {
                    "answer": answer,
                    "sources": result.get("sources", []) if result else [],
                    "key_points": result.get("key_points", []) if result else [],
                },
            },
        )

    except Exception as e:
        logger.error(f"Document query error: {e}")
        await send_error(websocket, str(e))


async def websocket_endpoint(websocket: WebSocket, dataset_id: str = Query(...)):
    dataset_id = dataset_id.strip('"')

    if dataset_id not in active_connections:
        active_connections[dataset_id] = []
    active_connections[dataset_id].append(websocket)

    try:
        await websocket.accept()
        logger.info(f"WebSocket connected for dataset: {dataset_id}")

        while True:
            data = await websocket.receive_text()

            try:
                payload = json.loads(data)
                query = payload.get("query", "")

                if not query:
                    await send_error(websocket, "Query is required")
                    continue

                csv_engine = get_csv_engine()
                doc_engine = get_document_engine()

                if csv_engine.dataset_exists(dataset_id):
                    await process_csv_query(websocket, dataset_id, query)
                elif doc_engine.dataset_exists(dataset_id):
                    await process_document_query(websocket, dataset_id, query)
                else:
                    await send_error(websocket, f"Dataset {dataset_id} not found")

            except json.JSONDecodeError:
                await send_error(websocket, "Invalid JSON payload")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for dataset: {dataset_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await send_error(websocket, str(e))
    finally:
        if dataset_id in active_connections:
            active_connections[dataset_id].remove(websocket)
