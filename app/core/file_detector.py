import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class FileType(str, Enum):
    DOCUMENT = "document"
    CSV = "csv"
    UNSUPPORTED = "unsupported"


SUPPORTED_EXTENSIONS = {
    ".pdf": FileType.DOCUMENT,
    ".docx": FileType.DOCUMENT,
    ".doc": FileType.DOCUMENT,
    ".csv": FileType.CSV,
}


MAGIC_BYTES = {
    b"%PDF": FileType.DOCUMENT,
    b"PK": FileType.DOCUMENT,
}


def detect_file_type(filename: str, file_bytes: bytes = None) -> FileType:
    ext = "." + filename.lower().split(".")[-1] if "." in filename else ""

    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]

    if file_bytes and len(file_bytes) >= 4:
        for magic, file_type in MAGIC_BYTES.items():
            if file_bytes[: len(magic)] == magic:
                if file_type == FileType.DOCUMENT:
                    return FileType.DOCUMENT

        if _is_csv_content(file_bytes):
            return FileType.CSV

    return FileType.UNSUPPORTED


def _is_csv_content(file_bytes: bytes) -> bool:
    try:
        text = file_bytes[:1024].decode("utf-8", errors="ignore")
        lines = text.split("\n")[:5]

        if len(lines) < 2:
            return False

        comma_counts = [line.count(",") for line in lines if line.strip()]
        if not comma_counts:
            return False

        avg_commas = sum(comma_counts) / len(comma_counts)
        return avg_commas >= 2

    except Exception:
        return False


def validate_file_size(file_bytes: bytes, max_size_mb: int = 100) -> tuple[bool, str]:
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"

    if len(file_bytes) == 0:
        return False, "Empty file"

    return True, "OK"


def get_file_type_name(file_type: FileType) -> str:
    return file_type.value
