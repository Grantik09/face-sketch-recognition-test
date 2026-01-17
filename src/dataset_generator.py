"""Generate synthetic dataset for demonstration"""
import numpy as np
from PIL import Image, ImageDraw
import random
from pathlib import Path
from src.face_embeddings import FaceEmbeddingModel

class SyntheticFaceGenerator:
    """Generate synthetic face images"""
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
    
    def generate_face(self, seed=None):
        """Generate a synthetic face image"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create blank image
        img = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Random face position and size
        face_x = random.randint(30, self.image_size[0] - 130)
        face_y = random.randint(20, self.image_size[1] - 140)
        face_size = random.randint(80, 130)
        
        # Draw face shape
        face_color = (random.randint(180, 240), random.randint(140, 200), random.randint(100, 160))
        draw.ellipse(
            [face_x, face_y, face_x + face_size, face_y + face_size + 20],
            fill=face_color,
            outline='black',
            width=2
        )
        
        # Draw eyes
        eye_y = face_y + face_size // 3
        left_eye_x = face_x + face_size // 4
        right_eye_x = face_x + 3 * face_size // 4
        eye_size = face_size // 6
        
        draw.ellipse(
            [left_eye_x - eye_size, eye_y - eye_size, left_eye_x + eye_size, eye_y + eye_size],
            fill='white',
            outline='black',
            width=2
        )
        draw.ellipse(
            [right_eye_x - eye_size, eye_y - eye_size, right_eye_x + eye_size, eye_y + eye_size],
            fill='white',
            outline='black',
            width=2
        )
        
        # Draw pupils
        pupil_size = eye_size // 2
        draw.ellipse(
            [left_eye_x - pupil_size, eye_y - pupil_size, left_eye_x + pupil_size, eye_y + pupil_size],
            fill='black'
        )
        draw.ellipse(
            [right_eye_x - pupil_size, eye_y - pupil_size, right_eye_x + pupil_size, eye_y + pupil_size],
            fill='black'
        )
        
        # Draw nose
        nose_x = face_x + face_size // 2
        nose_y = face_y + face_size // 2
        draw.polygon(
            [(nose_x, nose_y), (nose_x - face_size // 10, nose_y + face_size // 8),
             (nose_x + face_size // 10, nose_y + face_size // 8)],
            fill=face_color,
            outline='black'
        )
        
        # Draw mouth
        mouth_y = face_y + 2 * face_size // 3
        mouth_width = face_size // 3
        draw.arc(
            [nose_x - mouth_width, mouth_y, nose_x + mouth_width, mouth_y + face_size // 8],
            0, 180,
            fill='black',
            width=2
        )
        
        # Add some features variations
        if random.random() > 0.5:
            # Add hair
            draw.rectangle(
                [face_x, face_y - face_size // 4, face_x + face_size, face_y + face_size // 5],
                fill=(random.randint(50, 150), random.randint(30, 100), random.randint(0, 80)),
                outline='black'
            )
        
        return np.array(img)

class DatasetGenerator:
    """Generate dataset with embeddings"""
    
    def __init__(self, output_dir="data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.generator = SyntheticFaceGenerator()
        self.embedding_model = FaceEmbeddingModel("Facenet512")
    
    def generate_dataset(self, num_persons=50, images_per_person=1):
        """
        Generate synthetic dataset
        Args:
            num_persons: number of different people
            images_per_person: images per person
        """
        embeddings_dict = {}
        metadata_dict = {}
        
        first_names = ["John", "Emma", "Michael", "Sarah", "David", "Jennifer", "Robert", "Jessica",
                      "James", "Patricia", "Joseph", "Barbara", "Thomas", "Mary", "Charles", "Linda"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                     "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson"]
        
        categories = ["wanted", "missing", "suspect", "witness", "victim"]
        
        print(f"Generating {num_persons} synthetic faces...")
        
        for person_idx in range(num_persons):
            person_id = f"person_{person_idx:04d}"
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            name = f"{first_name} {last_name}"
            age = random.randint(18, 80)
            category = random.choice(categories)
            
            # Generate images and embeddings
            person_embeddings = []
            
            for img_idx in range(images_per_person):
                # Generate face
                face_image = self.generator.generate_face(seed=person_idx * 1000 + img_idx)
                face_normalized = face_image.astype(np.float32) / 255.0
                
                # Extract embedding
                embedding = self.embedding_model.extract_embedding(face_normalized)
                person_embeddings.append(embedding)
                
                # Save image
                images_dir = self.output_dir / "faces"
                images_dir.mkdir(exist_ok=True)
                Image.fromarray(face_image).save(
                    images_dir / f"{person_id}_face_{img_idx}.png"
                )
            
            # Average embeddings
            avg_embedding = np.mean(person_embeddings, axis=0)
            embeddings_dict[person_id] = avg_embedding
            metadata_dict[person_id] = {
                'name': name,
                'age': age,
                'category': category,
                'description': f"Synthetic person {person_idx}"
            }
            
            if (person_idx + 1) % 10 == 0:
                print(f"  Generated {person_idx + 1}/{num_persons}")
        
        return embeddings_dict, metadata_dict
