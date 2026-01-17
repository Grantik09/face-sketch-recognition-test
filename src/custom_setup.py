import sys
from pathlib import Path
import logging
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DatasetLoader
from src.database import EmbeddingDatabase
from src.face_embeddings import FaceEmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(dataset_path: str):
    print("=" * 70)
    print("Face Sketch Recognition System - Custom Dataset Setup")
    print("=" * 70)

    loader = DatasetLoader(dataset_path)
    dataset = loader.load_dataset()

    db = EmbeddingDatabase("data/embeddings.db")
    model = FaceEmbeddingModel("Facenet512")

    success = 0
    failed = 0

    for item in dataset:
        image_id = item["image_id"]
        image_path = item["image_path"]
        metadata = item["metadata"]

        # ✅ Add image_path to metadata so UI can display it later
        metadata["image_path"] = image_path

        try:
            embedding = model.extract_embedding(image_path)

            if embedding is None or not isinstance(embedding, np.ndarray):
                raise ValueError("Invalid embedding")

            db.add_person(image_id, embedding, metadata)
            success += 1
            print(f"✓ Added {image_id}")

        except Exception as e:
            failed += 1
            print(f"✗ Failed {image_id}: {e}")

    print("=" * 70)
    print(f"Processed: {success}")
    print(f"Failed: {failed}")
    print("Database: data/embeddings.db")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/custom_setup.py <dataset_path>")
        sys.exit(1)

    main(sys.argv[1])
