import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from llm.groq_client import get_groq_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CompressedContext:
    text: str
    source_nodes: List[str]
    compression_ratio: float
    original_length: int
    compressed_length: int


class ContextCompressor:
    def __init__(self, max_context_length: int = 8000):
        self.max_context_length = max_context_length

    def compress(
        self,
        query: str,
        results: List[Dict[str, Any]],
        max_context_length: Optional[int] = None,
    ) -> CompressedContext:
        max_len = max_context_length or self.max_context_length

        if not results:
            return CompressedContext(
                text="",
                source_nodes=[],
                compression_ratio=0.0,
                original_length=0,
                compressed_length=0,
            )

        original_texts = []
        source_nodes = []

        for r in results:
            if r.get("node"):
                text = r["node"].text if hasattr(r["node"], "text") else str(r["node"])
                node_id = r.get(
                    "node_id", r["node"].id if hasattr(r["node"], "id") else "unknown"
                )
            else:
                text = str(r.get("text", ""))
                node_id = r.get("node_id", "unknown")

            original_texts.append(text)
            source_nodes.append(node_id)

        combined_text = "\n\n".join(original_texts)
        original_length = len(combined_text)
        original_words = len(combined_text.split())

        if original_words <= max_len // 2:
            return CompressedContext(
                text=combined_text,
                source_nodes=source_nodes,
                compression_ratio=1.0,
                original_length=original_length,
                compressed_length=len(combined_text),
            )

        compressed_text = self._query_aware_compress(query, combined_text, max_len)

        compressed_length = len(compressed_text)

        compression_ratio = (
            compressed_length / original_length if original_length > 0 else 0
        )

        return CompressedContext(
            text=compressed_text,
            source_nodes=source_nodes,
            compression_ratio=compression_ratio,
            original_length=original_length,
            compressed_length=compressed_length,
        )

    def _query_aware_compress(self, query: str, context: str, max_length: int) -> str:
        try:
            groq = get_groq_client()

            prompt = f"""Given the following context and query, create a comprehensive summary that preserves ALL important details.

Query: {query}

Context:
{context}

Instructions:
1. Preserve ALL factual information, names, dates, numbers, and specific details
2. Keep technical terms, technologies, and methodologies mentioned
3. Include accomplishments, achievements, and quantifiable results
4. Maintain specific project names and descriptions
5. Preserve qualifications, skills, and competencies
6. Do NOT omit details just to make it shorter
7. Output comprehensive content that answers the query thoroughly
8. Output ONLY the compressed context, no explanations

Comprehensive Context:"""

            compressed = groq.generate(prompt, max_tokens=4000)

            if compressed and len(compressed.strip()) > 100:
                logger.info(
                    f"LLM compression: {len(context)} -> {len(compressed)} chars"
                )
                return compressed.strip()

        except Exception as e:
            logger.warning(f"LLM compression failed: {e}, using fallback")

        return self._fallback_compress(context, max_length)

    def _fallback_compress(self, text: str, max_length: int) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)

        compressed_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_length + sentence_words <= max_length // 2:
                compressed_sentences.append(sentence)
                current_length += sentence_words
            else:
                break

        if not compressed_sentences:
            words = text.split()
            compressed_sentences = [" ".join(words[: max_length // 2])]

        return " ".join(compressed_sentences)

    def compress_batch(
        self, query: str, results: List[Dict[str, Any]], batch_size: int = 10
    ) -> List[CompressedContext]:
        compressed_contexts = []

        for i in range(0, len(results), batch_size):
            batch = results[i : i + batch_size]
            compressed = self.compress(query, batch)
            compressed_contexts.append(compressed)

        return compressed_contexts

    def merge_compressed(
        self, contexts: List[CompressedContext], max_length: int = None
    ) -> CompressedContext:
        if not contexts:
            return CompressedContext(
                text="",
                source_nodes=[],
                compression_ratio=0.0,
                original_length=0,
                compressed_length=0,
            )

        merged_text_parts = []
        all_source_nodes = []
        total_original = 0
        total_compressed = 0

        max_len = max_length or self.max_context_length

        for ctx in contexts:
            all_source_nodes.extend(ctx.source_nodes)
            total_original += ctx.original_length
            total_compressed += ctx.compressed_length

            if (
                len(" ".join(merged_text_parts).split()) + len(ctx.text.split())
                <= max_len // 2
            ):
                merged_text_parts.append(ctx.text)

        merged_text = " ".join(merged_text_parts)

        return CompressedContext(
            text=merged_text,
            source_nodes=all_source_nodes,
            compression_ratio=total_compressed / total_original
            if total_original > 0
            else 0,
            original_length=total_original,
            compressed_length=len(merged_text),
        )


_compressor_instance: Optional[ContextCompressor] = None


def get_compressor() -> ContextCompressor:
    global _compressor_instance
    if _compressor_instance is None:
        _compressor_instance = ContextCompressor()
    return _compressor_instance
