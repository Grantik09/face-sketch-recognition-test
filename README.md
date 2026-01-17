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

<img width="488" height="691" alt="image" src="https://github.com/user-attachments/assets/d54c2af9-c6d6-4868-a7f0-885ae92f4892" />


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
