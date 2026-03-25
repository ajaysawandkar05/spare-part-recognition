# 🔍 HWL Spare Parts Intelligence (CLIP + DINOv2)

Advanced multi-modal spare parts retrieval system using:

- 🔵 CLIP → semantic + text understanding  
- 🟣 DINOv2 → fine-grained visual matching  

This hybrid system enables accurate industrial part recognition using:
- Visual similarity (DINOv2 + CLIP)
- Text similarity (CLIP)
- Metadata-aware retrieval

---

## 🚀 Features

- 📸 Image-based search (DINOv2 + CLIP fusion)
- 🧠 Text-based search (CLIP)
- 🔄 Multi-modal search (image + text)
- 🎯 Fine-grained matching using DINOv2
- ⚡ FastAPI backend
- 💻 React frontend
- 📊 Confidence scoring
- 🧩 Part-type filtering

---

## 🧱 Tech Stack

| Layer    | Tech |
|----------|------|
| Backend  | FastAPI · Python |
| Models   | CLIP (ViT-B/32) + DINOv2 (ViT-B/14) |
| Frontend | React · Vite |
| Data     | Image dataset + JSON + NumPy index |
| Deploy   | Docker (optional) |

---

## 🧠 System Architecture

### Image Query
```
Image → DINOv2 (70%) + CLIP (30%) → Visual Score
```

### Text Query
```
Text → CLIP → Text Score
```

### Final Score
```
Image + Text → Weighted fusion → Final Ranking
```

---

## 🚀 Quick Start

### 1. Clone Project

```bash
git clone <your-repo-url>
cd Spare_Part_Recognition
```

---

### 2. Prepare Dataset (IMPORTANT)

⚠️ Dataset is NOT included.

Add images manually:

```
data/hwl_images/
    <material_id>/
        image_1.png
        image_2.png
```

Example:

```
data/hwl_images/59-1014230-00867/image_1.png
```

❗ Without images → system will NOT work

---

### 3. Generate Descriptions

```bash
python scripts/generate_descriptions.py
```

Creates:

```
data/hwl_descriptions.json
```

Used by:
```
clip_engine.py
```

---

### 4. Install Backend

```bash
cd backend
pip install -r requirements.txt
```

---

### 5. Run Backend

```bash
uvicorn main:app --reload --port 8000
```

👉 First run:
- Loads CLIP + DINOv2
- Builds index automatically

---

### 6. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

---

### 7. Open App

- Frontend → http://localhost:5173  
- Backend → http://localhost:8000  
- API Docs → http://localhost:8000/docs  

---

## 🔍 How It Works

### Stage 1 — Feature Extraction
- CLIP → semantic features
- DINOv2 → fine-grained visual features

---

### Stage 2 — Retrieval

For image queries:
- DINOv2 similarity (70%)
- CLIP similarity (30%)

---

### Stage 3 — Text Matching

- CLIP text embeddings
- Matches descriptions

---

### Stage 4 — Fusion

```
Final Score = visual_weight * visual + text_weight * text
```

---

### Stage 5 — Ranking

- Max pooling per material
- Top-K results returned

---

## 📊 Confidence Logic

```python
confidence = "low" if (top1 - top2) < (top1 * 0.15) else "high"
```

---

## 📡 API Reference

| Method | Endpoint               | Description |
|--------|------------------------|------------|
| POST   | `/api/query`           | Search parts |
| GET    | `/api/materials`       | List materials |
| GET    | `/api/material/{id}`   | Material details |
| GET    | `/api/image/{id}`      | Get image |
| POST   | `/api/index/rebuild`   | Rebuild index |
| GET    | `/api/health`          | Health check |
| GET    | `/docs`                | Swagger UI |

---

## 📁 Project Structure

```
Spare_Part_Recognition/
├── backend/
│   ├── main.py
│   ├── clip_engine.py   ← CLIP + DINOv2 logic
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css       Full design system (industrial dark theme)
│   │   ├── main.jsx
│   │   └── components/
│   │       ├── Header.jsx
│   │       ├── SearchPanel.jsx   Drag-drop upload + text query + weights
│   │       ├── StatsBar.jsx      Match stats + query mode indicator
│   │       ├── ResultsGrid.jsx   Skeleton loaders + results layout
│   │       └── ResultCard.jsx    Image + score bars + warehouse metadata
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── .env.example
├── scripts/
│   ├── generate_descriptions.py
│   ├── extract_xlsx.py (optional)
│   └── evaluate.py
├── data/
│   ├── hwl_images/
│   ├── hwl_multi_images/
│   ├── hwl_dataset.json
│   ├── hwl_descriptions.json
│   └── hwl_clip_index.npz
├── photos_query/
├── docker-compose.yml
└── README.md
```

---

## ⚠️ Important Notes

- Dataset images are NOT included
- Must manually add images
- Descriptions required before running
- Index builds automatically
- No rotation augmentation (for accuracy)

---

## 🔥 Key Improvements (v5)

- ✅ Added DINOv2 (fine-grained vision)
- ✅ Hybrid scoring (DINOv2 + CLIP)
- ✅ Max pooling (better than averaging)
- ✅ Safer augmentations (no rotation)
- ✅ Confidence detection
- ✅ Cleaner ranking (no sigmoid)

---

## 🔮 Future Improvements

- Background removal
- Fine-tuning CLIP/DINO
- Object detection
- Better filtering
- UI improvements

---

## 👨‍💻 Author

Ajay  
Computer Engineering Student  

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!