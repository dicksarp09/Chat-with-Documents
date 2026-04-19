import logging
import time
from typing import List, Optional, Union
import numpy as np

from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        model_name: str = None,
        output_dimensionality: int = None,
    ):
        self.model_name = model_name or settings.embedding_model
        self.api_key = settings.gemini_api_key
        self._client = None
        self._output_dimensionality = output_dimensionality or settings.embedding_dimension
        self._embedding_dim = self._output_dimensionality

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                logger.warning("GEMINI_API_KEY not set. Embeddings will use fallback mode.")
                return None
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
                self._client = None
        return self._client

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 10,  # Free tier: 100/min, use smaller batches
        show_progress: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        texts = [t if t else " " for t in texts]

        if self.client is None:
            logger.warning("Gemini client unavailable, returning zero embeddings (fallback mode)")
            return np.zeros((len(texts), self._embedding_dim), dtype=np.float32)

        try:
            config = None
            if self._output_dimensionality:
                config = {"outputDimensionality": self._output_dimensionality}

            all_embeddings = []
            for start in range(0, len(texts), batch_size):
                end = min(start + batch_size, len(texts))
                batch = texts[start:end]

                for attempt in range(3):
                    try:
                        response = self.client.models.embed_content(
                            model=self.model_name,
                            contents=batch,
                            config=config
                        )
                        break
                    except Exception as rate_err:
                        if "RESOURCE_EXHAUSTED" in str(rate_err) and attempt < 2:
                            wait_time = 35 if attempt == 0 else 60
                            logger.warning(f"Rate limit hit, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        raise

                batch_embeddings = [list(e.values) for e in response.embeddings]
                all_embeddings.extend(batch_embeddings)

                if show_progress:
                    logger.info(f"Encoded {end}/{len(texts)} texts")

                # Rate limit: free tier is 100 req/min = 1.67/sec
                # With batch_size=10, need delay between batches
                if end < len(texts) and len(texts) > batch_size:
                    time.sleep(0.6)  # ~100 requests per minute

            embeddings = np.array(all_embeddings, dtype=np.float32)

            return embeddings

        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
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
        _embedder_instance = Embedder()
    return _embedder_instance


def encode_texts(texts: Union[str, List[str]]) -> np.ndarray:
    embedder = get_embedder()
    return embedder.encode(texts)
