import logging
import uuid
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DocumentEngineWrapper:
    def __init__(self):
        self.registry: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        self._sqlite_store = None
        self._try_load_from_sqlite()

    def _ensure_init(self):
        if not self._initialized:
            self._init_engine()
            self._initialized = True

    def _try_load_from_sqlite(self):
        """Load previously processed documents from SQLite."""
        try:
            from storage.sqlite_store import get_sqlite_store

            self._sqlite_store = get_sqlite_store()

            # Load all datasets at startup
            datasets = self._sqlite_store.get_all_datasets()
            if datasets:
                logger.info(f"Found {len(datasets)} previously processed datasets")
                for ds in datasets:
                    chunks = self._sqlite_store.load_dataset(ds["id"])
                    if chunks:
                        logger.info(f"  {ds['name']}: {len(chunks)} chunks ready")
        except Exception as e:
            logger.warning(f"Could not load from SQLite: {e}")
            self._sqlite_store = None

    def _init_engine(self):
        try:
            from parsers.pdf_parser import parse_pdf
            from parsers.docx_parser import parse_docx
            from chunking.hierarchical_chunker import chunk_document
            from embeddings.embedder import get_embedder
            from storage.vector_store import get_vector_store
            from retrieval.hybrid_retriever import get_hybrid_retriever
            from retrieval.reranker import get_reranker
            from compression.compressor import get_compressor
            from llm.reasoning import get_reasoning_pipeline
            from validation.json_validator import get_validator

            self._parse_pdf = parse_pdf
            self._parse_docx = parse_docx
            self._chunk_document = chunk_document
            self._get_embedder = get_embedder
            self._get_vector_store = get_vector_store
            self._get_hybrid_retriever = get_hybrid_retriever
            self._get_reranker = get_reranker
            self._get_compressor = get_compressor
            self._get_reasoning = get_reasoning_pipeline
            self._get_validator = get_validator

            logger.info("Document engine initialized")
        except ImportError as e:
            logger.warning(f"Some document engine components not available: {e}")

    def process(self, file_bytes: bytes, filename: str, **kwargs) -> Dict[str, Any]:
        self._ensure_init()

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=Path(filename).suffix, delete=False
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            if filename.lower().endswith(".docx"):
                parsed_doc = self._parse_pdf(tmp_path)
            elif filename.lower().endswith(".pdf"):
                parsed_doc = self._parse_pdf(tmp_path)
            else:
                raise ValueError(f"Unsupported document type: {filename}")

            structure = self._chunk_document(parsed_doc)

            vector_store = self._get_vector_store()
            logger.info(f"Vector store before add: {vector_store.get_stats()}")
            vector_store.add_nodes(structure.nodes)
            logger.info(f"Vector store after add: {vector_store.get_stats()}")

            # Save to SQLite for persistence
            if self._sqlite_store:
                chunks_data = [
                    {
                        "id": node.id,
                        "doc_id": doc_id,
                        "text": node.text,
                        "section_title": node.section_title,
                        "level": node.level,
                        "parent_id": node.parent_id,
                    }
                    for node in structure.nodes
                ]
                self._sqlite_store.save_dataset(
                    doc_id, filename, "document", chunks_data
                )

                # Save embeddings if available
                from embeddings.embedder import get_embedder

                embedder = get_embedder()
                texts = [node.text for node in structure.nodes]
                embeddings = embedder.encode(texts)
                chunk_emb = {
                    node.id: emb for node, emb in zip(structure.nodes, embeddings)
                }
                self._sqlite_store.save_embeddings(chunk_emb)

            hybrid_retriever = self._get_hybrid_retriever()
            hybrid_retriever.rebuild_index(doc_id)

            self.registry[doc_id] = {
                "doc_id": doc_id,
                "filename": filename,
                "file_path": tmp_path,
                "parsed_doc": parsed_doc,
                "structure": structure,
                "status": "processed",
            }

            return {
                "doc_id": doc_id,
                "filename": filename,
                "type": "document",
                "chunk_count": structure.total_chunks,
                "status": "processed",
                "message": "Document processed and saved to SQLite",
            }

        except Exception as e:
            logger.error(f"Document processing error: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def query(self, query: str, doc_id: str, **kwargs) -> Dict[str, Any]:
        self._ensure_init()

        if doc_id not in self.registry:
            return {
                "answer": f"Document {doc_id} not found",
                "summary": "",
                "key_points": [],
                "risks": [],
                "obligations": [],
                "actions": [],
                "sources": [],
            }

        try:
            hybrid_retriever = self._get_hybrid_retriever()
            reranker = self._get_reranker()
            compressor = self._get_compressor()
            reasoning = self._get_reasoning()
            validator = self._get_validator()

            logger.info(
                f"Hybrid retriever corpus size: {len(hybrid_retriever.corpus_ids)}"
            )

            # Try without doc_id first to see if retrieval works at all
            retrieved_all = hybrid_retriever.retrieve(query, doc_id=None, top_k=5)
            logger.info(f"Retrieved {len(retrieved_all)} results (no filter)")

            retrieved = hybrid_retriever.retrieve(query, doc_id=doc_id, top_k=20)
            logger.info(f"Retrieved {len(retrieved)} results (with doc_id filter)")

            if not retrieved and retrieved_all:
                logger.info("No filtered results, using unfiltered results")
                retrieved = retrieved_all

            if not retrieved:
                return {
                    "answer": "No relevant content found",
                    "summary": "",
                    "key_points": [],
                    "risks": [],
                    "obligations": [],
                    "actions": [],
                    "sources": [],
                }

            reranked = reranker.rerank(query, retrieved, top_k=5)
            compressed = compressor.compress(query, reranked)
            query_result = reasoning.query_analysis(query, compressed)

            validation = validator.validate_with_fallback(
                query_result.model_dump(), "query"
            )

            if validation.is_valid:
                return validation.data

            return query_result.model_dump()

        except Exception as e:
            logger.error(f"Document query error: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "summary": "",
                "key_points": [],
                "risks": [],
                "obligations": [],
                "actions": [],
                "sources": [],
            }

    def list_documents(self) -> list:
        return [
            {
                "doc_id": doc_id,
                "filename": data["filename"],
                "status": data["status"],
            }
            for doc_id, data in self.registry.items()
        ]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if doc_id not in self.registry:
            return None
        data = self.registry[doc_id]
        return {
            "doc_id": doc_id,
            "filename": data["filename"],
            "status": data["status"],
        }

    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self.registry:
            file_path = self.registry[doc_id]["file_path"]
            if os.path.exists(file_path):
                os.unlink(file_path)
            del self.registry[doc_id]
            return True
        return False


_document_engine_instance: Optional[DocumentEngineWrapper] = None


def get_document_engine() -> DocumentEngineWrapper:
    global _document_engine_instance
    if _document_engine_instance is None:
        _document_engine_instance = DocumentEngineWrapper()
    return _document_engine_instance
