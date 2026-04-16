"""
SQLite-backed vector store for persistence.
Stores embeddings, BM25 index metadata, and dataset info.
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "documents.db"


class SQLiteStore:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.conn = None
        self._ensure_connected()
        self._init_schema()

    def _ensure_connected(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row

    def _init_schema(self):
        self._ensure_connected()
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                created_at TEXT,
                chunk_count INTEGER,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                dataset_id TEXT,
                doc_id TEXT,
                text TEXT,
                section_title TEXT,
                level INTEGER,
                parent_id TEXT,
                created_at TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_dataset 
            ON chunks(dataset_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc
            ON chunks(doc_id)
        """)

        self.conn.commit()
        logger.info(f"SQLite schema initialized at {self.db_path}")

    def save_dataset(
        self,
        dataset_id: str,
        name: str,
        doc_type: str,
        chunks_data: List[Dict],
        metadata: Dict = None,
    ) -> bool:
        """Save dataset with all chunks and embeddings to SQLite."""
        self._ensure_connected()
        cursor = self.conn.cursor()

        import datetime

        now = datetime.datetime.now().isoformat()

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO datasets 
                (id, name, type, created_at, chunk_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    dataset_id,
                    name,
                    doc_type,
                    now,
                    len(chunks_data),
                    json.dumps(metadata or {}),
                ),
            )

            for chunk in chunks_data:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (id, dataset_id, doc_id, text, section_title, level, parent_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk.get("id", chunk.get("node_id", "")),
                        dataset_id,
                        chunk.get("doc_id", ""),
                        chunk.get("text", ""),
                        chunk.get("section_title", ""),
                        chunk.get("level", 3),
                        chunk.get("parent_id"),
                        now,
                    ),
                )

            self.conn.commit()
            logger.info(f"Saved dataset {dataset_id}: {len(chunks_data)} chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            return False

    def load_dataset(self, dataset_id: str) -> Optional[List[Dict]]:
        """Load all chunks for a dataset from SQLite."""
        self._ensure_connected()
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT id, dataset_id, doc_id, text, section_title, level, parent_id
            FROM chunks 
            WHERE dataset_id = ?
        """,
            (dataset_id,),
        )

        rows = cursor.fetchall()
        if not rows:
            return None

        chunks = []
        for row in rows:
            chunks.append(
                {
                    "id": row["id"],
                    "dataset_id": row["dataset_id"],
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "section_title": row["section_title"],
                    "level": row["level"],
                    "parent_id": row["parent_id"],
                }
            )

        logger.info(f"Loaded dataset {dataset_id}: {len(chunks)} chunks")
        return chunks

    def get_all_datasets(self) -> List[Dict]:
        """Get all datasets."""
        self._ensure_connected()
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT id, name, type, created_at, chunk_count, metadata
            FROM datasets
            ORDER BY created_at DESC
        """)

        rows = cursor.fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "created_at": r["created_at"],
                "chunk_count": r["chunk_count"],
                "metadata": json.loads(r["metadata"] or "{}"),
            }
            for r in rows
        ]

    def save_embeddings(self, chunk_embeddings: Dict[str, np.ndarray]) -> bool:
        """Save embeddings for chunks."""
        self._ensure_connected()
        cursor = self.conn.cursor()

        import datetime

        now = datetime.datetime.now().isoformat()

        try:
            for chunk_id, embedding in chunk_embeddings.items():
                blob = embedding.astype(np.float32).tobytes()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO embeddings (chunk_id, embedding)
                    VALUES (?, ?)
                """,
                    (chunk_id, blob),
                )

            self.conn.commit()
            logger.info(f"Saved {len(chunk_embeddings)} embeddings")
            return True

        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return False

    def load_embeddings(self, chunk_ids: List[str]) -> Dict[str, np.ndarray]:
        """Load embeddings for given chunk IDs."""
        self._ensure_connected()
        cursor = self.conn.cursor()

        if not chunk_ids:
            return {}

        placeholders = ",".join(["?"] * len(chunk_ids))
        cursor.execute(
            f"""
            SELECT chunk_id, embedding
            FROM embeddings
            WHERE chunk_id IN ({placeholders})
        """,
            chunk_ids,
        )

        rows = cursor.fetchall()
        embeddings = {}
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            embeddings[row["chunk_id"]] = embedding

        logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its data."""
        self._ensure_connected()
        cursor = self.conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE dataset_id = ?)",
                (dataset_id,),
            )
            cursor.execute("DELETE FROM chunks WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            self.conn.commit()
            logger.info(f"Deleted dataset {dataset_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete dataset: {e}")
            return False

    def get_stats(self) -> Dict:
        self._ensure_connected()
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM datasets")
        dataset_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM chunks")
        chunk_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM embeddings")
        embed_count = cursor.fetchone()["count"]

        return {
            "datasets": dataset_count,
            "chunks": chunk_count,
            "embeddings": embed_count,
            "db_path": self.db_path,
        }

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


def get_sqlite_store() -> SQLiteStore:
    global _sqlite_store
    if _sqlite_store is None:
        _sqlite_store = SQLiteStore()
    return _sqlite_store


_sqlite_store: Optional[SQLiteStore] = None
