"""Setup script to generate dataset and embeddings"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_generator import DatasetGenerator
from src.database import EmbeddingDatabase
import numpy as np

def setup():
    """Generate dataset and save to database"""
    print("=" * 60)
    print("Face Sketch Recognition System - Setup")
    print("=" * 60)
    
    # Generate dataset
    print("\n1. Generating synthetic dataset...")
    generator = DatasetGenerator("data")
    embeddings_dict, metadata_dict = generator.generate_dataset(num_persons=50, images_per_person=2)
    print(f"   ✓ Generated {len(embeddings_dict)} persons")
    
    # Save to database
    print("\n2. Saving to database...")
    db = EmbeddingDatabase("data/embeddings.db")
    
    for person_id, metadata in metadata_dict.items():
        db.add_person(
            person_id,
            metadata['name'],
            metadata['age'],
            metadata['category'],
            metadata['description']
        )
        
        # Save embedding
        embedding = embeddings_dict[person_id]
        db.add_embedding(person_id, embedding)
    
    print(f"   ✓ Saved {len(metadata_dict)} persons to database")
    
    # Save embeddings as numpy arrays for faster loading
    print("\n3. Caching embeddings...")
    embeddings_dir = Path("data/embeddings")
    embeddings_dir.mkdir(exist_ok=True)
    
    for person_id, embedding in embeddings_dict.items():
        np.save(
            embeddings_dir / f"{person_id}.npy",
            embedding.astype(np.float32)
        )
    
    print(f"   ✓ Cached {len(embeddings_dict)} embeddings")
    
    print("\n" + "=" * 60)
    print("Setup complete! Run: streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    setup()
