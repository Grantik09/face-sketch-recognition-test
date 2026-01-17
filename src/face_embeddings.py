import numpy as np
import cv2
import logging
from deepface import DeepFace
from PIL import Image
import io

logger = logging.getLogger(__name__)


class FaceEmbeddingModel:
    # ✅ class-level cache (shared across Streamlit reruns)
    _cached_model = None
    _cached_model_name = None

    def __init__(self, model_name="Facenet512"):
        self.model_name = model_name

        # ✅ Load model only once
        if FaceEmbeddingModel._cached_model is None or FaceEmbeddingModel._cached_model_name != model_name:
            logger.info(f"Loading DeepFace model once: {model_name}")
            FaceEmbeddingModel._cached_model = DeepFace.build_model(model_name)
            FaceEmbeddingModel._cached_model_name = model_name

        self.model = FaceEmbeddingModel._cached_model
        logger.info(f"Face embedding model ready: {model_name}")

    def _to_rgb_numpy(self, image_source):
        """Convert input to RGB numpy image."""
        if isinstance(image_source, str):
            img_bgr = cv2.imread(image_source)
            if img_bgr is None:
                raise ValueError(f"Cannot read image: {image_source}")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if isinstance(image_source, Image.Image):
            return np.array(image_source.convert("RGB"))

        if isinstance(image_source, np.ndarray):
            if image_source.ndim == 2:
                return cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB)
            return image_source

        if isinstance(image_source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_source)).convert("RGB")
            return np.array(img)

        raise TypeError(f"Unsupported image type: {type(image_source)}")

    def extract_embedding(self, image_source):
        """
        Extract embedding using DeepFace.represent safely
        (works in Streamlit reruns without duplicate-layer errors)
        """
        try:
            img_rgb = self._to_rgb_numpy(image_source)

            result = DeepFace.represent(
                img_path=img_rgb,
                model_name=self.model_name,
                model=self.model,              # ✅ reuse loaded model
                enforce_detection=False
            )

            if result is None:
                raise ValueError("DeepFace returned None")

            embedding = None

            # list output
            if isinstance(result, list) and len(result) > 0:
                first = result[0]
                if isinstance(first, dict) and "embedding" in first:
                    embedding = first["embedding"]
                elif isinstance(first, (float, int, np.floating, np.integer)):
                    if len(result) > 10:
                        embedding = result
                    else:
                        raise ValueError("DeepFace returned invalid float list")
                elif isinstance(first, np.ndarray):
                    embedding = first.tolist()
                else:
                    raise ValueError(f"Unexpected list element type: {type(first)}")

            # dict output
            elif isinstance(result, dict):
                embedding = result.get("embedding")

            # direct array output
            elif isinstance(result, np.ndarray):
                embedding = result.tolist()

            else:
                raise ValueError(f"Unexpected DeepFace output type: {type(result)}")

            if embedding is None:
                raise ValueError("Embedding missing")

            emb = np.array(embedding, dtype=np.float32)

            if emb.ndim != 1:
                raise ValueError(f"Embedding must be 1D, got shape: {emb.shape}")

            if emb.size < 100:
                raise ValueError(f"Embedding too small: {emb.size}")

            if np.all(emb == 0):
                raise ValueError("Embedding all zeros")

            return emb

        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
