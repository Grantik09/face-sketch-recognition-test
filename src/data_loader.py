import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Dataset format:
    image.png
    image.txt with EXACT keys:

    name:
    age:
    gender:
    category:
    description:
    """

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    REQUIRED_FIELDS = {"name", "age", "gender", "category", "description"}

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        if not self.dataset_path.is_dir():
            raise ValueError("Dataset path must be a directory")

        logger.info(f"Dataset loader initialized for: {dataset_path}")

    def get_image_text_pairs(self) -> List[Tuple[str, str, str]]:
        pairs = []

        image_files = [
            f for f in self.dataset_path.iterdir()
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

        logger.info(f"Found {len(image_files)} image files")

        for img in image_files:
            txt = self.dataset_path / f"{img.stem}.txt"
            if not txt.exists():
                raise FileNotFoundError(f"Missing metadata file for {img.name}")

            pairs.append((str(img), str(txt), img.stem))

        logger.info(f"Found {len(pairs)} complete image-text pairs")
        return pairs

    def parse_text_metadata(self, text_path: str) -> Dict[str, str]:
        metadata = {}

        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue

                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                metadata[key] = value

        missing = self.REQUIRED_FIELDS - metadata.keys()
        if missing:
            raise ValueError(
                f"{text_path} is missing required fields: {missing}"
            )

        return {
            "name": metadata["name"],
            "age": metadata["age"],
            "gender": metadata["gender"],
            "category": metadata["category"],
            "description": metadata["description"],
        }

    def load_dataset(self) -> List[Dict]:
        dataset = []

        for img_path, txt_path, image_id in self.get_image_text_pairs():
            metadata = self.parse_text_metadata(txt_path)

            dataset.append({
                "image_path": img_path,
                "image_id": image_id,
                "metadata": metadata
            })

        logger.info(f"Successfully loaded {len(dataset)} items")
        return dataset
