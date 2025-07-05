# app.py

import os
import fitz  # PyMuPDF
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

print("🔑 Loaded KEY:", os.getenv("OPENAI_API_KEY"))


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Function to load text from all PDFs in 'docs/' folder ---
def load_all_pdfs(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(folder_path, filename)) as doc:
                for page in doc:
                    all_text += page.get_text()
    return all_text

# --- Function to split text into chunks ---
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

# --- Function to get sentence embeddings ---
@st.cache_data
def get_embeddings(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = model.encode(text_chunks)
    return model, vectors

# --- Function to store vectors in FAISS index ---
def build_faiss_index(vectors):
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))
    return index

# --- Streamlit Web App Starts Here ---
st.title("📚 Ask Dr. Elbert’s Research Papers")

# Load and process PDFs
pdf_text = load_all_pdfs("docs")
chunks = split_text(pdf_text)
model, vectors = get_embeddings(chunks)
index = build_faiss_index(vectors)

# User input
query = st.text_input("Ask a question about Dr. Elbert's research:")

if query:
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=5)
    context = "\n\n".join([chunks[i] for i in I[0]])

    # Use OpenAI API to generate answer
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert on Dr. Elbert's research."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
    )

    st.write("### Answer:")
    st.write(response.choices[0].message.content)

    with st.expander("🔍 Retrieved Context"):
        st.write(context)
