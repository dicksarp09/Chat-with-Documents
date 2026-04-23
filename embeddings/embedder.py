import logging
import threading
from typing import List, Optional, Union
import numpy as np

from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    _instance_lock = threading.Lock()
    _warm_client = None
    _warm_dimension = None

    def __init__(
        self,
        model_name: str = None,
        output_dimensionality: int = None,
    ):
        self.model_name = model_name or settings.embedding_model
        self._output_dimensionality = output_dimensionality or settings.embedding_dimension
        self._embedding_dim = self._output_dimensionality
        self._client = None

        if Embedder._warm_client is not None and Embedder._warm_dimension == self._output_dimensionality:
            logger.info("Using pre-warmed embedding model")
            self._client = Embedder._warm_client
            self._embedding_dim = Embedder._warm_dimension
            self._output_dimensionality = Embedder._warm_dimension

    @property
    def client(self):
        if self._client is None:
            with Embedder._instance_lock:
                if Embedder._warm_client is not None:
                    self._client = Embedder._warm_client
                    self._embedding_dim = Embedder._warm_dimension
                    return self._client

                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Loading embedding model: {self.model_name}")
                    self._client = SentenceTransformer(self.model_name)
                    actual_dim = self._client.get_sentence_embedding_dimension()
                    self._embedding_dim = actual_dim
                    self._output_dimensionality = actual_dim

                    Embedder._warm_client = self._client
                    Embedder._warm_dimension = actual_dim

                    logger.info(f"Embedding model loaded (dimension: {actual_dim})")
                except Exception as e:
                    logger.error(f"Failed to load embedding model: {e}")
                    raise
        return self._client

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        texts = [t if t else " " for t in texts]

        if len(texts) == 0:
            return np.array([], dtype=np.float32)

        try:
            embeddings = self.client.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
                prompt_name=None,
                prompt=None,
            )

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return np.zeros((len(texts), self._embedding_dim), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return float(np.dot(embedding1, embedding2))

    def similarity_batch(
        self, query_embedding: np.ndarray, document_embeddings: np.ndarray
    ) -> np.ndarray:
        return np.dot(document_embeddings, query_embedding)


_embedder_instance: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        with threading.Lock():
            if _embedder_instance is None:
                _embedder_instance = Embedder()
    return _embedder_instance


def encode_texts(texts: Union[str, List[str]]) -> np.ndarray:
    embedder = get_embedder()
    return embedder.encode(texts)


def warmup_embedder():
    """Pre-warm the embedder to avoid first-query latency."""
    embedder = get_embedder()
    embedder.encode(["warmup text"], batch_size=1)
    logger.info("Embedder warmed up")
