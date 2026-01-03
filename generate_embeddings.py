"""
Generate vector embeddings for ICD-10 codes
"""
import sqlite3
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

print("=" * 80)
print("GENERATING VECTOR EMBEDDINGS")
print("=" * 80)

# Load embedding model
print("\n1. Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded")

# Load ICD-10 codes from database
print("\n2. Loading ICD-10 codes from database...")
conn = sqlite3.connect('real_medical_codes.db')
cursor = conn.cursor()
cursor.execute("SELECT code, description FROM icd10_codes")
rows = cursor.fetchall()
conn.close()

codes = [row[0] for row in rows]
descriptions = [row[1] for row in rows]
print(f"✓ Loaded {len(codes):,} ICD-10 codes")

# Generate embeddings
print("\n3. Generating embeddings (this may take 5-10 minutes)...")
embeddings = model.encode(descriptions, show_progress_bar=True)
print(f"✓ Generated {len(embeddings):,} embeddings")
print(f"✓ Embedding dimensions: {embeddings.shape[1]}D")

# Save to pickle file
print("\n4. Saving to vector_db.pkl...")
data = {
    "embeddings": embeddings,
    "codes": codes,
    "descriptions": descriptions
}

with open("vector_db.pkl", "wb") as f:
    pickle.dump(data, f)

print(f"✓ Saved vector database")
print(f"✓ File size: {len(pickle.dumps(data)) / 1024 / 1024:.1f} MB")

print("\n" + "=" * 80)
print("VECTOR DATABASE READY!")
print("=" * 80)
print("\nYou can now run the backend:")
print("  cd backend")
print("  python main.py")
