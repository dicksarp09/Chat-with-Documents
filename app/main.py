import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.upload import router as upload_router
from api.routes.query import router as query_router
from api.routes.websocket import websocket_endpoint
from core.config import settings
from embeddings.embedder import get_embedder
from storage.vector_store import get_vector_store

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.version}")

    try:
        logger.info("Initializing embedding model...")
        embedder = get_embedder()
        logger.info(f"Embedder ready, dimension: {embedder.get_embedding_dim()}")
    except Exception as e:
        logger.warning(f"Embedder initialization warning: {e}")

    try:
        logger.info("Initializing vector store...")
        vector_store = get_vector_store()
        logger.info("Vector store ready")
    except Exception as e:
        logger.warning(f"Vector store initialization warning: {e}")

    logger.info("Application startup complete")

    yield

    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Unified Document and CSV Intelligence Engine - single ingestion gateway",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api/v1", tags=["Unified Upload"])
app.include_router(query_router, prefix="/api/v1", tags=["Unified Query"])


@app.websocket("/ws/query/{dataset_id}")
async def ws_query(websocket: WebSocket, dataset_id: str):
    await websocket_endpoint(websocket, dataset_id)


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.version,
        "status": "running",
        "endpoints": {
            "upload": "/api/v1/upload",
            "query": "/api/v1/query",
            "datasets": "/api/v1/datasets",
            "datasets/{id}": "/api/v1/datasets/{dataset_id}",
            "websocket": "/ws/query/{dataset_id}",
        },
        "flow": {
            "1_upload": "POST /api/v1/upload (auto-routes to document or csv engine)",
            "2_query": "POST /api/v1/query?dataset_id=X&q=your question",
            "3_list": "GET /api/v1/datasets",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.app_name}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500, content={"detail": str(exc), "type": type(exc).__name__}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
