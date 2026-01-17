# Quick Start Guide - Face Sketch Recognition System

## 5-Minute Setup

### 1. Install

```bash
# Clone repository
git clone <repo-url>
cd face-sketch-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Dataset

```bash
python scripts/setup.py
```

Wait for dataset generation (~2-3 minutes on first run)

### 3. Run App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser

## Using the App

### Sketch Recognition

1. Click **Sketch Recognition** tab
2. Upload a sketch image
3. View top matches with confidence scores

### Face Matching

1. Click **Face Matching** tab
2. Upload a face photo
3. See similar faces in database

### Database

1. Click **Database** tab
2. Search for specific persons
3. View all persons and statistics

## Configuration

### Sidebar Settings

- **Number of matches:** 1-10 (default: 5)
- **Confidence threshold:** 0-100% (default: 50%)
- **Matching method:** cosine, euclidean, or combined

📁 Project Structure
face-sketch-recognition-test/
│
├── app_custom.py
├── venv/
│
├── data/
│   └── embeddings.db
│
├── dataset/
│   ├── 01349_1_F.png
│   ├── 01349_1_F.txt
│   ├── 01350_1_F.png
│   ├── 01350_1_F.txt
│   └── ...
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── face_embeddings.py
    ├── database.py
    ├── matching.py
    ├── preprocessing.py
    └── custom_setup.py
## Tips

✓ Use clear, detailed sketches for better results
✓ Lower confidence threshold for more matches
✓ Try different matching methods
✓ Check database statistics in sidebar

## Common Issues

| Issue               | Solution                                    |
| ------------------- | ------------------------------------------- |
| "Database is empty" | Run `python scripts/setup.py`               |
| No matches found    | Lower confidence threshold                  |
| Slow performance    | Reduce number of matches                    |
| Module not found    | Run `pip install -r requirements.txt` again |

## Next Steps

- **Add Your Data:** Replace synthetic data in `scripts/setup.py`
- **Deploy:** Use `streamlit cloud` for free hosting
- **Customize:** Edit `app.py` to match your needs

## Detailed Documentation

See `IMPLEMENTATION_GUIDE.md` for comprehensive documentation

---

**Happy Face Matching!** 🎉
