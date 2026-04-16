import logging
from typing import Dict, Any, List, Optional
import fitz
import uuid
import re

from schemas.output_schema import SectionData, ParsedDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PdfParser:
    def __init__(self):
        self.heading_patterns = [
            r"^[A-Z][A-Z\s\d\-]+$",
            r"^\d+\.\s+[A-Z]",
            r"^[IVX]+\.\s+[A-Z]",
            r"^[A-Z][a-z]+(\s[A-Z][a-z]+)*:$",
        ]

    def _is_heading(self, text: str, font_size: float, font_name: str = "") -> bool:
        if not text or len(text.strip()) < 3:
            return False

        if len(text) > 200:
            return False

        if font_size >= 12:
            for pattern in self.heading_patterns:
                if re.match(pattern, text.strip()):
                    return True

        if "bold" in font_name.lower():
            return True

        return False

    def _extract_text_from_page(self, page) -> List[Dict[str, Any]]:
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []

        for block in blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_blocks.append(
                            {
                                "text": span["text"],
                                "font": span.get("font", ""),
                                "size": span.get("size", 0),
                                "bbox": span.get("bbox", []),
                            }
                        )

        return text_blocks

    def _group_into_paragraphs(
        self, blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        paragraphs = []
        current_para = {"text": "", "blocks": []}

        for block in blocks:
            text = block["text"].strip()
            if not text:
                if current_para["text"]:
                    paragraphs.append(current_para)
                    current_para = {"text": "", "blocks": []}
                continue

            if self._is_heading(text, block["size"], block["font"]):
                if current_para["text"]:
                    paragraphs.append(current_para)
                    current_para = {"text": "", "blocks": []}
                paragraphs.append(
                    {
                        "text": text,
                        "is_heading": True,
                        "size": block["size"],
                        "font": block["font"],
                        "blocks": [block],
                    }
                )
            else:
                if current_para.get("is_heading"):
                    paragraphs.append(current_para)
                    current_para = {"text": "", "blocks": [], "is_heading": False}
                current_para["text"] += " " + text
                current_para["blocks"].append(block)

        if current_para["text"]:
            paragraphs.append(current_para)

        for para in paragraphs:
            para["text"] = para["text"].strip()
            if "is_heading" not in para:
                para["is_heading"] = False

        return paragraphs

    def _detect_heading_level(
        self, text: str, size: float, all_paragraphs: List
    ) -> int:
        max_size = max([p.get("size", 0) for p in all_paragraphs], default=16)

        if size >= max_size - 2:
            return 1
        elif size >= max_size - 5:
            return 2
        else:
            return 3

    def _build_section_tree(
        self, paragraphs: List[Dict[str, Any]]
    ) -> List[SectionData]:
        sections = []
        current_section = None
        current_content = []

        for para in paragraphs:
            text = para["text"]
            is_heading = para.get("is_heading", False)
            level = para.get("level", 1)

            if is_heading:
                if current_section:
                    current_section.content = " ".join(current_content)
                    if current_content:
                        sections.append(current_section)

                current_section = SectionData(
                    title=text, level=level, content="", children=[]
                )
                current_content = []
            else:
                if current_section:
                    current_content.append(text)
                else:
                    current_content.append(text)

        if current_section and current_content:
            current_section.content = " ".join(current_content)
            sections.append(current_section)
        elif current_content:
            section = SectionData(
                title="", level=1, content=" ".join(current_content), children=[]
            )
            sections.append(section)

        for section in sections:
            if section.level == 1 and section.content:
                child_paragraphs = self._split_into_subsections(section.content)
                if child_paragraphs:
                    for child_text in child_paragraphs:
                        if child_text.strip():
                            child = SectionData(
                                title="",
                                level=2,
                                content=child_text.strip(),
                                children=[],
                            )
                            section.children.append(child)
                section.content = ""

        return sections

    def _split_into_subsections(self, text: str, max_length: int = 500) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        subsections = []
        current = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_length // 2 and current:
                if current:
                    subsections.append(" ".join(current))
                    current = []
                    current_length = 0
            current.append(sentence)
            current_length += sentence_length

        if current:
            subsections.append(" ".join(current))

        return subsections

    def _extract_tables(self, page) -> List[Dict[str, Any]]:
        tables = []
        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            if block["type"] == 1:
                bbox = block["bbox"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if height > 20 and width > 50:
                    tables.append({"bbox": bbox, "type": "image/table"})

        return tables

    def parse(self, file_path: str) -> ParsedDocument:
        logger.info(f"Parsing PDF file: {file_path}")

        doc_id = str(uuid.uuid4())
        filename = file_path.split("/")[-1].split("\\")[-1]

        all_paragraphs = []

        try:
            doc = fitz.open(file_path)
            page_count = len(doc)

            for page_num in range(page_count):
                page = doc[page_num]
                blocks = self._extract_text_from_page(page)
                page_paragraphs = self._group_into_paragraphs(blocks)

                for para in page_paragraphs:
                    para["page"] = page_num + 1

                all_paragraphs.extend(page_paragraphs)

            doc.close()
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise

        for i, para in enumerate(all_paragraphs):
            if para.get("is_heading"):
                level = self._detect_heading_level(
                    para["text"], para.get("size", 0), all_paragraphs
                )
                para["level"] = level

        sections = self._build_section_tree(all_paragraphs)

        full_text = "\n\n".join(
            [
                s.content if s.content else " ".join([c.content for c in s.children])
                for s in sections
            ]
        )

        metadata = {
            "page_count": page_count,
            "paragraph_count": len(all_paragraphs),
            "heading_count": len([p for p in all_paragraphs if p.get("is_heading")]),
        }

        logger.info(f"Extracted {len(sections)} sections from PDF")

        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type="pdf",
            sections=sections,
            full_text=full_text,
            metadata=metadata,
        )


def parse_pdf(file_path: str) -> ParsedDocument:
    parser = PdfParser()
    return parser.parse(file_path)
