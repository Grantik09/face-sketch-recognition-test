"""
Streamlit Web Interface for Face Sketch Recognition
Uses custom dataset loaded from user's folder
Improved UI to display results properly and show detailed person information
"""

import streamlit as st
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import logging

# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database import EmbeddingDatabase
from face_embeddings import FaceEmbeddingModel
from matching import FaceMatcher
from preprocessing import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Face Sketch Recognition System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Initialize Streamlit session state
# -----------------------
for key in ['db_loaded', 'db', 'model', 'matcher']:
    if key not in st.session_state:
        st.session_state[key] = None

st.session_state.db_loaded = st.session_state.db_loaded or False

# -----------------------
# Function to load database and model
# -----------------------
def load_database():
    """Load database and model on app start"""
    if not st.session_state.db_loaded:
        with st.spinner("Loading database and model... This may take a moment on first run."):
            try:
                st.session_state.db = EmbeddingDatabase("data/embeddings.db")
                st.session_state.model = FaceEmbeddingModel("Facenet512")
                st.session_state.matcher = FaceMatcher(st.session_state.db)
                st.session_state.db_loaded = True
                return True
            except Exception as e:
                st.error(f"Failed to load database: {str(e)}")
                st.info("Run setup first: python src/custom_setup.py /path/to/your/dataset")
                return False
    return True

# Load database before continuing
if not load_database():
    st.stop()

db = st.session_state.db
model = st.session_state.model
matcher = st.session_state.matcher

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
<style>
.match-card {
    background-color: #f0f2f6;
    border-left: 4px solid #1f77e4;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}
.match-card.high { border-left-color: #28a745; }
.match-card.medium { border-left-color: #ffc107; }
.match-card.low { border-left-color: #dc3545; }
.confidence-score { font-size: 24px; font-weight: bold; margin: 10px 0; }
.confidence-high { color: #28a745; }
.confidence-medium { color: #ffc107; }
.confidence-low { color: #dc3545; }
.match-name { font-size: 18px; font-weight: bold; margin: 5px 0; }
.match-info { font-size: 14px; color: #555; margin: 3px 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Helper Functions
# -----------------------
def get_confidence_color(confidence):
    if confidence >= 0.75:
        return "high"
    elif confidence >= 0.55:
        return "medium"
    else:
        return "low"

def display_match_results(matches, confidence_threshold):
    """Display matches in improved card layout"""
    if not matches:
        st.warning("No matches found in database.")
        return

    filtered = [m for m in matches if m['confidence'] >= confidence_threshold]
    if not filtered:
        st.warning(f"No matches with confidence >= {confidence_threshold:.0%}")
        st.info("Try lowering the confidence threshold or uploading a different image.")
        return

    st.success(f"✓ Found {len(filtered)} match(es)")
    st.divider()

    for match in filtered:
        confidence = match['confidence']
        color_class = get_confidence_color(confidence)
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.metric("Confidence", f"{confidence:.1%}", delta=None)

        with col2:
            st.markdown(f"<div class='match-name'>👤 {match['name']}</div>", unsafe_allow_html=True)
            info_text = ""
            if match.get('category') and match['category'] != 'Unknown':
                info_text += f"**Category:** {match['category']} | "
            if match.get('age') and match['age'] != 'N/A':
                info_text += f"**Age:** {match['age']} | "
            if match.get('description'):
                info_text += f"**Info:** {match['description'][:60]}..."
            if info_text:
                st.markdown(info_text)

        with col3:
            if confidence >= 0.75:
                st.success("High Match")
            elif confidence >= 0.55:
                st.warning("Medium Match")
            else:
                st.info("Low Match")
        st.divider()

# -----------------------
# Sidebar Configuration
# -----------------------
embeddings_dict, metadata_dict = db.get_all_embeddings()
total_persons = len(embeddings_dict)

st.sidebar.title("⚙️ Configuration")
num_matches = st.sidebar.slider("Number of matches to return:", 1, 20, 5)
confidence_threshold = st.sidebar.slider("Confidence threshold:", 0.0, 1.0, 0.50, step=0.05)
matching_method = st.sidebar.selectbox("Matching method:", ["cosine", "euclidean", "combined"])
st.sidebar.divider()
st.sidebar.subheader("Database Info")
st.sidebar.metric("Total Persons", total_persons)

# -----------------------
# Main Tabs
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["Sketch Recognition", "Face Matching", "Database", "Help"])

# -------- Tab 1: Sketch Recognition --------
with tab1:
    st.header("Sketch-Based Recognition")
    col1, col2 = st.columns([1,1])
    with col1:
        uploaded_sketch = st.file_uploader("Choose a sketch image", type=["jpg","jpeg","png","bmp"], key="sketch")
        if uploaded_sketch:
            sketch_image = Image.open(uploaded_sketch)
            st.image(sketch_image, caption="Uploaded Sketch", use_column_width=True)
    with col2:
        if uploaded_sketch:
            st.subheader("Processing Results")
            preprocessor = ImagePreprocessor()
            processed_image = preprocessor.preprocess(sketch_image)
            with st.spinner("Extracting facial features..."):
                embedding = model.extract_embedding(sketch_image)
            if embedding is not None and np.linalg.norm(embedding) > 0:
                with st.spinner("Searching database..."):
                    matches = matcher.find_matches(embedding, top_k=num_matches, method=matching_method)
                stats = matcher.compute_statistics(embedding)
                with st.expander("Debug Info"):
                    st.json(stats)
                display_match_results(matches, confidence_threshold)
                # Download results
                if matches:
                    results_text = "Face Sketch Recognition Results\n" + "="*60 + "\n\n"
                    for idx, m in enumerate(matches,1):
                        results_text += f"{idx}. {m['name']}\n"
                        results_text += f"   Confidence: {m['confidence']:.1%}\n"
                        results_text += f"   Category: {m.get('category','Unknown')}\n"
                        if m.get('description'):
                            results_text += f"   Description: {m['description']}\n"
                        results_text += "\n"
                    st.download_button("📥 Download Results", results_text, "recognition_results.txt", "text/plain")
            else:
                st.error("Could not extract face from image. Please try another image with a clear face.")

# -------- Tab 2: Face Matching --------
with tab2:
    st.header("Face-to-Face Matching")
    col1, col2 = st.columns([1,1])
    with col1:
        uploaded_face = st.file_uploader("Choose a face image", type=["jpg","jpeg","png","bmp"], key="face")
        if uploaded_face:
            face_image = Image.open(uploaded_face)
            st.image(face_image, caption="Uploaded Photo", use_column_width=True)
    with col2:
        if uploaded_face:
            st.subheader("Processing Results")
            with st.spinner("Processing..."):
                embedding = model.extract_embedding(face_image)
            if embedding is not None and np.linalg.norm(embedding) > 0:
                with st.spinner("Finding matches..."):
                    matches = matcher.find_matches(embedding, top_k=num_matches, method=matching_method)
                display_match_results(matches, confidence_threshold)
            else:
                st.error("Could not extract face from image.")

# -------- Tab 3: Database --------
with tab3:
    st.header("Database Management")
    if total_persons > 0:
        st.metric("Total Persons in Database", total_persons)
        display_data = []
        for pid, meta in metadata_dict.items():
            display_data.append({
                "ID": pid[:20],
                "Name": meta.get("name","Unknown"),
                "Age": meta.get("age","N/A"),
                "Category": meta.get("category","Unknown"),
                "Description": meta.get("description","N/A")[:50]
            })
        st.dataframe(display_data, use_container_width=True, hide_index=True)
        st.subheader("Search Database")
        search_name = st.text_input("Search by name:")
        if search_name:
            filtered = [p for p in display_data if search_name.lower() in p["Name"].lower()]
            if filtered:
                st.dataframe(filtered, use_container_width=True, hide_index=True)
            else:
                st.info(f"No persons found matching '{search_name}'")
    else:
        st.warning("Database is empty. Run setup first!")
        st.code("python src/custom_setup.py /path/to/your/dataset")

# -------- Tab 4: Help --------
with tab4:
    st.header("Help & Documentation")
    st.subheader("Understanding Confidence Scores")
    st.markdown("""
    - **75%+**: Excellent match (likely the same person)
    - **55-75%**: Good match (moderate confidence)
    - **40-55%**: Possible match (use with caution)
    - **<40%**: Weak match (likely not a match)
    """)
    st.subheader("Tips for Best Results")
    st.markdown("""
    1. **Image Quality**: Use clear, well-lit photos
    2. **Face Visibility**: Ensure face is clearly visible
    3. **Image Size**: Larger images work better
    4. **Threshold Settings**: Lower = more matches, Higher = stricter
    5. **Matching Method**: Cosine recommended, Euclidean or Combined optional
    """)
    st.subheader("Troubleshooting")
    with st.expander("No matches found"):
        st.markdown("""
        - Try lowering confidence threshold
        - Use different matching method
        - Check if database is loaded
        - Upload a clearer image
        """)
    with st.expander("Getting started"):
        st.markdown("""
        1. Prepare your dataset with images and .txt metadata files
        2. Run setup: `python src/custom_setup.py /path/to/dataset`
        3. Launch app: `streamlit run app_custom.py`
        4. Upload images in the tabs above
        """)

st.divider()
st.caption("Face Sketch Recognition System v2.0 | Improved Accuracy & UI")
