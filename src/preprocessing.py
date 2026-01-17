"""Image preprocessing utilities for face sketch recognition"""

from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import io
from streamlit.runtime.uploaded_file_manager import UploadedFile


class ImagePreprocessor:
    """Handles image preprocessing for face matching"""

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def load_image(self, image_source):
        """Load image from various sources (file path, PIL Image, numpy array, bytes, Streamlit UploadedFile)"""

        # Streamlit UploadedFile
        if isinstance(image_source, UploadedFile):
            image = Image.open(image_source).convert("RGB")
            return np.array(image)

        # File path
        if isinstance(image_source, (str, Path)):
            image = cv2.imread(str(image_source))
            if image is None:
                raise ValueError(f"Could not load image from {image_source}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # PIL Image
        if isinstance(image_source, Image.Image):
            return np.array(image_source.convert("RGB"))

        # NumPy array
        if isinstance(image_source, np.ndarray):
            if image_source.ndim == 2:
                return cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB)
            return image_source

        # Raw bytes
        if isinstance(image_source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_source)).convert("RGB")
            return np.array(image)

        # ❌ Unsupported type
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    def resize_image(self, image, size=None):
        """Resize image to target size"""
        if size is None:
            size = self.target_size
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def normalize_image(self, image):
        """Normalize image to [0, 1] range"""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image

    def enhance_contrast(self, image):
        """Enhance image contrast for better face detection"""
        if image.ndim == 3:
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return enhanced.astype(np.float32) / 255.0

        return image

    def preprocess_sketch(self, sketch_image):
        """Preprocess sketch image for matching"""
        image = self.load_image(sketch_image)
        image = self.resize_image(image)
        image = image.astype(np.float32) / 255.0
        image = self.enhance_contrast(image)
        return image

    def preprocess_face(self, face_image):
        image = self.load_image(face_image)
        image = self.resize_image(image)
        # DO NOT normalize
        return image

    def extract_face_roi(self, image, face_coords):
        """Extract ROI from face coordinates (x, y, w, h)"""
        x, y, w, h = face_coords
        return image[y:y + h, x:x + w]

    def normalize_sketch(self, sketch):
        """Apply special normalization for sketches"""
        if sketch.ndim == 3:
            sketch = cv2.cvtColor((sketch * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            sketch = (sketch * 255).astype(np.uint8)

        sketch = cv2.equalizeHist(sketch)
        return sketch.astype(np.float32) / 255.0


class SketchEnhancer:
    """Enhance sketch quality for better matching"""

    @staticmethod
    def denoise(sketch_image):
        sketch = (sketch_image * 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(
            sketch, h=10, templateWindowSize=7, searchWindowSize=21
        )
        return denoised.astype(np.float32) / 255.0

    @staticmethod
    def sharpen(sketch_image):
        sketch = (sketch_image * 255).astype(np.uint8)
        kernel = np.array(
            [[-1, -1, -1],
             [-1,  9, -1],
             [-1, -1, -1]],
            dtype=np.float32
        )
        sharpened = cv2.filter2D(sketch, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.float32) / 255.0

    @staticmethod
    def edge_detection(sketch_image):
        sketch = (sketch_image * 255).astype(np.uint8)
        edges = cv2.Canny(sketch, 50, 150)
        return edges.astype(np.float32) / 255.0
