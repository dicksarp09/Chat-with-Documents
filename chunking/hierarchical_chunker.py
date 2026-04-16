import logging
import uuid
from typing import List, Dict, Any, Optional, Set

from schemas.output_schema import (
    ParsedDocument,
    SectionData,
    ChunkNode,
    DocumentStructure,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalChunker:
    def __init__(
        self, min_chunk_size: int = 200, max_chunk_size: int = 400, overlap: int = 50
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def _split_text_into_chunks(
        self,
        text: str,
        section_title: str = "",
        parent_id: Optional[str] = None,
        doc_id: str = "",
    ) -> List[ChunkNode]:
        if not text or not text.strip():
            return []

        words = text.split()
        if len(words) <= self.max_chunk_size:
            chunk_id = str(uuid.uuid4())
            return [
                ChunkNode(
                    id=chunk_id,
                    text=text.strip(),
                    level=3,
                    doc_id=doc_id,
                    section_title=section_title,
                    parent_id=parent_id,
                    children=[],
                    metadata={},
                )
            ]

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.max_chunk_size, len(words))

            if end < len(words) and end - start > self.min_chunk_size:
                while (
                    end > start + self.min_chunk_size
                    and words[end - 1][-1] not in ".!?,;:"
                ):
                    end -= 1

                if end == start + self.min_chunk_size:
                    end = min(start + self.max_chunk_size, len(words))

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunk_id = str(uuid.uuid4())
            chunk_node = ChunkNode(
                id=chunk_id,
                text=chunk_text.strip(),
                level=3,
                doc_id=doc_id,
                section_title=section_title,
                parent_id=parent_id,
                children=[],
                metadata={},
            )
            chunks.append(chunk_node)

            if end >= len(words):
                break

            start = end - self.overlap
            if start >= len(words):
                break

        return chunks

    def _process_section(
        self,
        section: SectionData,
        doc_id: str,
        parent_id: Optional[str] = None,
        level: int = 1,
    ) -> List[ChunkNode]:
        all_nodes = []

        section_id = str(uuid.uuid4())

        section_text = (
            f"{section.title}: {section.content}" if section.content else section.title
        )
        section_node = ChunkNode(
            id=section_id,
            text=section_text.strip(),
            level=level,
            doc_id=doc_id,
            section_title=section.title,
            parent_id=parent_id,
            children=[],
            metadata={"section_title": section.title, "original_level": section.level},
        )
        all_nodes.append(section_node)

        if section.content:
            content_chunks = self._split_text_into_chunks(
                section.content, section.title, section_id, doc_id
            )

            for chunk in content_chunks:
                chunk.parent_id = section_id
                all_nodes.append(chunk)
                section_node.children.append(chunk.id)

        for child_section in section.children:
            child_nodes = self._process_section(
                child_section, doc_id, section_id, level + 1
            )

            if child_nodes:
                child_section_node = child_nodes[0]
                child_section_node.parent_id = section_id
                all_nodes.extend(child_nodes)
                section_node.children.append(child_section_node.id)

        return all_nodes

    def _flatten_children(self, sections: List[SectionData]) -> List[SectionData]:
        flattened = []
        for section in sections:
            flattened.append(section)
            if section.children:
                flattened.extend(self._flatten_children(section.children))
        return flattened

    def chunk(self, parsed_doc: ParsedDocument) -> DocumentStructure:
        logger.info(f"Starting hierarchical chunking for document: {parsed_doc.doc_id}")

        all_nodes = []
        root_nodes = []

        if not parsed_doc.sections:
            root_id = str(uuid.uuid4())
            root_node = ChunkNode(
                id=root_id,
                text=parsed_doc.full_text,
                level=1,
                doc_id=parsed_doc.doc_id,
                section_title="Document Root",
                parent_id=None,
                children=[],
                metadata={},
            )
            all_nodes.append(root_node)

            chunks = self._split_text_into_chunks(
                parsed_doc.full_text, "Document Root", root_id, parsed_doc.doc_id
            )

            for chunk in chunks:
                chunk.parent_id = root_id
                all_nodes.append(chunk)
                root_node.children.append(chunk.id)
        else:
            for section in parsed_doc.sections:
                section_nodes = self._process_section(
                    section, parsed_doc.doc_id, None, section.level
                )
                all_nodes.extend(section_nodes)
                if section_nodes:
                    root_nodes.append(section_nodes[0])

        node_map = {node.id: node for node in all_nodes}

        for node in all_nodes:
            if node.parent_id and node.parent_id in node_map:
                parent = node_map[node.parent_id]
                if node.id not in parent.children:
                    parent.children.append(node.id)

        for node in all_nodes:
            node.children = list(set(node.children))

        structure = DocumentStructure(
            doc_id=parsed_doc.doc_id,
            nodes=all_nodes,
            total_chunks=len([n for n in all_nodes if n.level == 3]),
        )

        logger.info(
            f"Created {len(all_nodes)} total nodes, {structure.total_chunks} leaf chunks"
        )

        return structure

    def get_all_texts(self, structure: DocumentStructure) -> List[Dict[str, Any]]:
        texts = []
        for node in structure.nodes:
            texts.append(
                {
                    "id": node.id,
                    "text": node.text,
                    "level": node.level,
                    "doc_id": node.doc_id,
                    "section_title": node.section_title,
                    "parent_id": node.parent_id,
                }
            )
        return texts

    def get_related_nodes(
        self, node_id: str, structure: DocumentStructure, max_depth: int = 2
    ) -> List[ChunkNode]:
        node_map = {node.id: node for node in structure.nodes}

        if node_id not in node_map:
            return []

        related = []
        visited: Set[str] = set()

        def collect_related(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)

            if current_id in node_map:
                node = node_map[current_id]
                related.append(node)

                if node.parent_id:
                    collect_related(node.parent_id, depth + 1)

                for child_id in node.children:
                    collect_related(child_id, depth + 1)

        collect_related(node_id, 0)

        return related


def chunk_document(parsed_doc: ParsedDocument) -> DocumentStructure:
    chunker = HierarchicalChunker()
    return chunker.chunk(parsed_doc)
