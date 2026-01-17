"""Face matching and similarity computation with improved accuracy"""

import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceMatcher:
    """Match query sketch/face against database of faces with improved accuracy"""

    def __init__(self, database):
        self.db = database
        self.reload()

    def reload(self):
        """Reload embeddings + metadata from database"""
        self.embeddings_dict, self.metadata_dict = self.db.get_all_embeddings()
        self.person_ids = list(self.embeddings_dict.keys())
        self.embedding_matrix = (
            np.array(list(self.embeddings_dict.values()), dtype=np.float32)
            if self.embeddings_dict else np.array([], dtype=np.float32)
        )
        logger.info(f"Loaded {len(self.person_ids)} embeddings from database")

    def find_matches(self, query_embedding: np.ndarray,
                     top_k: int = 5,
                     method: str = "cosine") -> List[Dict]:

        # Refresh
        self.reload()

        if len(self.embedding_matrix) == 0:
            logger.warning("No embeddings in database")
            return []

        query_embedding = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        else:
            return []

        if method == "cosine":
            similarities = self._compute_cosine_similarities(query_embedding)
        elif method == "euclidean":
            similarities = self._compute_euclidean_similarities(query_embedding)
        else:
            similarities = self._compute_combined_similarities(query_embedding)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            person_id = self.person_ids[idx]
            confidence = float(similarities[idx])
            metadata = self.metadata_dict.get(person_id, {})

            results.append({
                "rank": rank,
                "person_id": person_id,
                "name": metadata.get("name", "Unknown"),
                "confidence": confidence,
                "category": metadata.get("category", "Unknown"),
                "description": metadata.get("description", ""),
                "age": metadata.get("age", "N/A"),
                "gender": metadata.get("gender", ""),
                "image_path": metadata.get("image_path", None),  # ✅ NEW
                "method": method
            })

        confidences = [f"{r['confidence']:.2%}" for r in results]
        logger.info(f"Found {len(results)} matches with confidences: {confidences}")

        return results

    def _compute_cosine_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        normalized_embeddings = self.embedding_matrix / norms

        raw_similarities = np.dot(normalized_embeddings, query_embedding)
        similarities = (raw_similarities + 1) / 2
        similarities = np.power(similarities, 1.1)

        return np.clip(similarities, 0, 1)

    def _compute_euclidean_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(self.embedding_matrix - query_embedding, axis=1)

        p95 = np.percentile(distances, 95)
        if p95 == 0:
            p95 = 1.0

        similarities = 1 - (distances / (p95 * 1.5))
        return np.clip(similarities, 0, 1)

    def _compute_combined_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        cosine_sim = self._compute_cosine_similarities(query_embedding)
        euclidean_sim = self._compute_euclidean_similarities(query_embedding)
        combined = 0.7 * cosine_sim + 0.3 * euclidean_sim
        return np.clip(combined, 0, 1)

    def compute_statistics(self, query_embedding: np.ndarray) -> Dict:
        if len(self.embedding_matrix) == 0:
            return {}

        query_embedding = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        else:
            return {}

        norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        normalized_matrix = self.embedding_matrix / norms

        cosine_sims = np.dot(normalized_matrix, query_embedding)

        return {
            "mean_similarity": float(np.mean(cosine_sims)),
            "max_similarity": float(np.max(cosine_sims)),
            "min_similarity": float(np.min(cosine_sims)),
            "std_similarity": float(np.std(cosine_sims)),
            "database_size": len(self.person_ids)
        }
