import logging
from typing import List, Optional, Union
import numpy as np
import torch

from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name or settings.embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize_embeddings
        self._model = None
        self._embedding_dim = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self._embedding_dim}")
        return self._model

    @property
    def embedding_dim(self):
        if self._embedding_dim is None:
            _ = self.model
        return self._embedding_dim

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        texts = [t if t else " " for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=convert_to_numpy,
            device=self.device,
        )

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        if self.normalize:
            return float(np.dot(embedding1, embedding2))
        else:
            return float(
                np.dot(embedding1, embedding2)
                / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            )

    def similarity_batch(
        self, query_embedding: np.ndarray, document_embeddings: np.ndarray
    ) -> np.ndarray:
        if self.normalize:
            return np.dot(document_embeddings, query_embedding)
        else:
            return np.dot(document_embeddings, query_embedding) / (
                np.linalg.norm(document_embeddings, axis=1)
                * np.linalg.norm(query_embedding)
            )


_embedder_instance: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance


def encode_texts(texts: Union[str, List[str]]) -> np.ndarray:
    embedder = get_embedder()
    return embedder.encode(texts)
