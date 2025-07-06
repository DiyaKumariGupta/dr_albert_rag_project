# setup_elbert_index.py

import os
import fitz
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def get_embeddings(chunks, model):
    return model.encode(chunks)

def build_faiss_index(vectors):
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))
    return index

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Collect and process PDFs
docs_dir = "docs"
all_text = ""
for filename in os.listdir(docs_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(docs_dir, filename)
        all_text += extract_text_from_pdf(file_path)

chunks = split_text(all_text)
vectors = get_embeddings(chunks, model)
index = build_faiss_index(vectors)

# Save index and chunks
faiss.write_index(index, "precomputed_index.index")
with open("elbert_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Precomputed index and chunks saved successfully.")
