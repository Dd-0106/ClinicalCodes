<<<<<<< HEAD
# AI Medical Coding System

Automated ICD-10 code generation using RAG AI. Generates precise medical codes from unstructured medical reports with 90%+ accuracy.

## Features

- 74,044 real ICD-10 codes from official CMS database
- Comprehensive symptom extraction across all medical departments
- Intelligent negation handling (NO VOMITING, NO FEVER)
- Multi-category support (Symptoms, Diagnoses, Procedures)
- React frontend + FastAPI backend
- 90%+ accuracy with confidence scores

## Prerequisites

- Python 3.8+
- Node.js 18+
- 4GB RAM

## Setup & Run

### 1. Generate Data Files (First Time Only)

The database and embeddings files are too large for GitHub. Generate them locally:

```bash
python scrape_real_codes.py
```

This will create:
- `real_medical_codes.db` (74,044 ICD-10 codes)
- `vector_db.pkl` (AI embeddings)

**Note**: This takes ~5-10 minutes and requires internet connection.

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend runs on: `http://localhost:8000`

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on: `http://localhost:5173`

### 3. Access Application

Open browser: `http://localhost:5173`

## Testing

```bash
python test_all_departments.py
```

## Project Structure

```
├── backend/
│   ├── main.py              # RAG AI engine & API
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main UI component
│   │   └── App.css
│   └── package.json
├── real_medical_codes.db    # 74,044 ICD-10 codes
├── vector_db.pkl            # AI embeddings
├── scrape_real_codes.py     # Data scraper
└── test_all_departments.py  # Tests
```

## API Usage

**POST** `/api/code`

```json
{
  "report_text": "PATIENT WITH FEVER AND COUGH"
}
```

**Response:**

```json
{
  "codes": [
    {
      "code": "R5081",
      "description": "Fever",
      "confidence": 87.3,
      "category": "symptom"
    }
  ],
  "total_codes": 2,
  "avg_confidence": 85.5
}
```

API docs: `http://localhost:8000/docs`

## Deployment (ngrok)

```bash
# Terminal 1: Start backend
cd backend
python main.py

# Terminal 2: Expose backend
ngrok http 8000

# Terminal 3: Start frontend
cd frontend
npm run dev

# Terminal 4: Expose frontend
ngrok http 5173
```

Update API URL in `frontend/src/App.jsx` with ngrok backend URL.

## Environment Variables

Create `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

## License

MIT

## Documentation

For detailed project information, see [PROJECT_REPORT.md](PROJECT_REPORT.md) which includes:
- Abstract and Introduction
- Methodology and Architecture
- Implementation Details
- Results and Performance Metrics
- Discussion and Future Work
- ngrok Deployment (Completed)
=======
# ClinicalCodes
>>>>>>> 16e3c44ca25fa60e42e0700d17486371da9ec210
