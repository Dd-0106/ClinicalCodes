# Quick Setup Guide

## First Time Setup

### 1. Clone Repository
```bash
git clone https://github.com/Dd-0106/ClinicalCodes.git
cd ClinicalCodes
```

### 2. Generate Data Files
```bash
python scrape_real_codes.py
```
This creates:
- `real_medical_codes.db` (74,044 ICD-10 codes)
- `vector_db.pkl` (AI embeddings)

**Time**: ~5-10 minutes

### 3. Setup Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### 4. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```

### 5. Access
Open: `http://localhost:5173`

## Environment Variables

Create `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

## Done!
Your AI medical coding system is ready! 🚀
