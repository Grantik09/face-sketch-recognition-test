import sqlite3
import json
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EmbeddingDatabase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Streamlit-safe sqlite connection
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                person_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def add_person(self, person_id: str, embedding: np.ndarray, metadata: dict):
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO embeddings (person_id, embedding, metadata)
            VALUES (?, ?, ?)
        """, (
            person_id,
            json.dumps(embedding.tolist()),
            json.dumps(metadata)
        ))

        self.conn.commit()

    def get_all_embeddings(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT person_id, embedding, metadata FROM embeddings")

        embeddings = {}
        metadata = {}

        for pid, emb, meta in cursor.fetchall():
            embeddings[pid] = np.array(json.loads(emb), dtype=np.float32)
            metadata[pid] = json.loads(meta)

        return embeddings, metadata

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
