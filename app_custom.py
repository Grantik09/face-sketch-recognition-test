"""
Streamlit Web Interface for Face Sketch Recognition
Uses structured dataset (image + structured .txt files)
Improved UI and full metadata handling
"""

import streamlit as st
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from face_embeddings import FaceEmbeddingModel
from database import EmbeddingDatabase
from matching import FaceMatcher

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Face Sketch Recognition",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for cards
st.markdown("""
<style>
.match-card {background-color: #f0f2f6; border-left: 4px solid #1f77e4; padding: 15px; margin:10px 0; border-radius:5px;}
.match-card.high {border-left-color: #28a745;}
.match-card.medium {border-left-color: #ffc107;}
.match-card.low {border-left-color: #dc3545;}
.confidence-score {font-size:24px; font-weight:bold; margin:10px 0;}
.confidence-high {color:#28a745;}
.confidence-medium {color:#ffc107;}
.confidence-low {color:#dc3545;}
.match-name {font-size:18px; font-weight:bold; margin:5px 0;}
.match-info {font-size:14px; color:#555; margin:3px 0;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False
    st.session_state.db = None
    st.session_state.model = None
    st.session_state.matcher = None


@st.cache_resource
def load_model():
    return FaceEmbeddingModel("Facenet512")

@st.cache_resource
def load_db():
    return EmbeddingDatabase(str(Path(__file__).parent / "data" / "embeddings.db"))

def load_database():
    try:
        st.session_state.db = load_db()
        st.session_state.model = load_model()
        st.session_state.matcher = FaceMatcher(st.session_state.db)
        return True
    except Exception as e:
        st.error(f"Failed to load database/model: {e}")
        return False


def get_confidence_color(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    elif confidence >= 0.55:
        return "medium"
    else:
        return "low"


def display_match_results(matches, confidence_threshold):
    """Display results with cards + matched images"""
    if not matches:
        st.warning("No matches found.")
        return

    filtered = [m for m in matches if m["confidence"] >= confidence_threshold]
    if not filtered:
        st.warning(f"No matches with confidence >= {confidence_threshold:.0%}")
        return

    st.success(f"✓ Found {len(filtered)} match(es)")
    st.divider()

    for match in filtered:
        confidence = match["confidence"]

        col1, col2, col3, col4 = st.columns([1, 2, 1, 1.2])

        # ✅ Confidence
        with col1:
            st.metric("Confidence", f"{confidence:.1%}")

        # ✅ Person details
        with col2:
            st.markdown(f"<div class='match-name'>👤 {match['name']}</div>", unsafe_allow_html=True)

            info_text = ""
            if match.get("category"):
                info_text += f"**Category:** {match['category']}  \n"
            if match.get("age"):
                info_text += f"**Age:** {match['age']}  \n"
            if match.get("gender"):
                info_text += f"**Gender:** {match['gender']}  \n"
            if match.get("description"):
                info_text += f"**Info:** {match['description']}"

            if info_text:
                st.markdown(info_text)

        # ✅ Status label
        with col3:
            if confidence >= 0.75:
                st.success("High Match")
            elif confidence >= 0.55:
                st.warning("Medium Match")
            else:
                st.info("Low Match")

        # ✅ Show matched image
        with col4:
            img_path = match.get("image_path")
            if img_path and Path(img_path).exists():
                matched_img = Image.open(img_path).convert("RGB")
                st.image(matched_img, caption="Matched Person", use_column_width=True)
            else:
                st.info("No image found")

        st.divider()


# -----------------------------
# MAIN APP
# -----------------------------
st.title("🔍 Face Sketch Recognition System")
st.subheader("Advanced Facial Recognition from Sketches & Photos")

if not load_database():
    st.stop()

# Sidebar
embeddings_dict, metadata_dict = st.session_state.db.get_all_embeddings()
total_persons = len(embeddings_dict)

st.sidebar.title("⚙️ Configuration")
num_matches = st.sidebar.slider("Number of matches", 1, 20, 5)
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
matching_method = st.sidebar.selectbox("Matching method", ["cosine", "euclidean", "combined"])
st.sidebar.metric("Total Persons", total_persons)

if total_persons == 0:
    st.warning("Database is empty. Please run setup first: python src/custom_setup.py <dataset_path>")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Sketch Recognition", "Face Matching", "Database", "Help"])

# -----------------------------
# TAB 1: Sketch Recognition
# -----------------------------
with tab1:
    st.header("Sketch Recognition")
    uploaded_sketch = st.file_uploader("Upload Sketch", type=["jpg", "jpeg", "png", "bmp"], key="sketch")

    if uploaded_sketch:
        sketch_image = Image.open(uploaded_sketch).convert("RGB")
        st.image(sketch_image, caption="Uploaded Sketch", use_column_width=True)

        with st.spinner("Extracting embedding..."):
            embedding = st.session_state.model.extract_embedding(sketch_image)

        if embedding is not None and np.linalg.norm(embedding) > 0:
            with st.spinner("Finding matches..."):
                matches = st.session_state.matcher.find_matches(
                    embedding, top_k=num_matches, method=matching_method
                )

            stats = st.session_state.matcher.compute_statistics(embedding)
            with st.expander("Debug Info"):
                st.json(stats)

            display_match_results(matches, confidence_threshold)
        else:
            st.error("Could not extract face from image. Try a clearer image.")

# -----------------------------
# TAB 2: Face Matching
# -----------------------------
with tab2:
    st.header("Face-to-Face Matching")
    uploaded_face = st.file_uploader("Upload Face Photo", type=["jpg", "jpeg", "png", "bmp"], key="face")

    if uploaded_face:
        face_image = Image.open(uploaded_face).convert("RGB")
        st.image(face_image, caption="Uploaded Photo", use_column_width=True)

        with st.spinner("Extracting embedding..."):
            embedding = st.session_state.model.extract_embedding(face_image)

        if embedding is not None and np.linalg.norm(embedding) > 0:
            with st.spinner("Finding matches..."):
                matches = st.session_state.matcher.find_matches(
                    embedding, top_k=num_matches, method=matching_method
                )
            display_match_results(matches, confidence_threshold)
        else:
            st.error("Could not extract face from image.")

# -----------------------------
# TAB 3: Database
# -----------------------------
with tab3:
    st.header("Database Records")
    if total_persons > 0:
        display_data = []
        for pid, meta in metadata_dict.items():
            display_data.append({
                "ID": pid,
                "Name": meta.get("name", "Unknown"),
                "Age": meta.get("age", ""),
                "Category": meta.get("category", ""),
                "Gender": meta.get("gender", ""),
                "Description": meta.get("description", "")[:50]
            })

        st.dataframe(display_data, use_container_width=True, hide_index=True)

        search_name = st.text_input("Search by name")
        if search_name:
            filtered = [p for p in display_data if search_name.lower() in p["Name"].lower()]
            if filtered:
                st.dataframe(filtered, use_container_width=True, hide_index=True)
            else:
                st.info(f"No persons found for '{search_name}'")
    else:
        st.warning("Database empty. Run setup first!")

# -----------------------------
# TAB 4: Help
# -----------------------------
with tab4:
    st.header("Help & Documentation")
    st.subheader("Metadata Format")
    st.markdown("""
Each image should have a .txt file with the following fields:

name: John Doe
age: 55
gender: M
category: suspect
description: Some info

""")

    st.subheader("Tips")
    st.markdown("""
1. Upload clear frontal images  
2. Confidence threshold affects results  
3. Use Cosine for general matching  
4. Lower threshold = more matches, higher = stricter  
""")