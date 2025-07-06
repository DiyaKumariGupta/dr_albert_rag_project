# app.py

# --- First imports ---
import streamlit as st
st.set_page_config(page_title="Dr. Elbert PDF Q&A", layout="wide")

# --- Other Imports ---
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import pickle
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor

# --- Teal Theme Styling ---
st.markdown("""
    <style>
        :root {
            --primary-color: #006d6d;
            --accent-color: #009999;
            --background-color: #e0f7f7;
            --text-color: #003333;
            --font-family: 'Segoe UI', sans-serif;
        }

        html, body, .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: var(--font-family);
        }

        .block-container {
            padding: 2rem;
        }

        .stTextInput > div > div > input {
            background-color: white;
            color: var(--text-color);
            border-radius: 8px;
            padding: 10px;
            border: 1px solid var(--accent-color);
        }

        .stDownloadButton > button, .stButton > button {
            background-color: var(--primary-color);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 16px;
            border: none;
        }

        .stDownloadButton > button:hover, .stButton > button:hover {
            background-color: var(--accent-color);
        }

        .stExpanderHeader {
            font-weight: bold;
            color: var(--primary-color) !important;
        }

        .stRadio > div {
            color: var(--text-color);
        }

        h1, h2, h3 {
            color: var(--primary-color);
        }

        .css-1v0mbdj, .css-10trblm {
            background-color: var(--background-color);
        }
    </style>
""", unsafe_allow_html=True)


# --- Load environment ---
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load embedding model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- Utility Functions ---
def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def extract_text_from_pdf(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        with ThreadPoolExecutor() as executor:
            pages = list(executor.map(lambda p: p.get_text(), doc))
    return "".join(pages)

def split_text(text, chunk_size=500, overlap=50):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def get_embeddings(chunks):
    return model.encode(chunks)

def build_faiss_index(vectors):
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))
    return index

@st.cache_resource
def load_elbert_data():
    try:
        index = faiss.read_index("precomputed_index.index")
        with open("elbert_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    except:
        return None, None

# --- App Header ---
st.title("📘 Ask Questions About Research PDFs")
st.markdown("Upload your own research paper or explore Dr. Elbert's to ask questions instantly.")

# --- Sidebar Source Selector ---
use_elbert = st.sidebar.radio("Choose document source:", ["Use Dr. Elbert's Papers", "Upload Your Own PDFs"])

# --- Document Handling ---
if use_elbert == "Upload Your Own PDFs":
    uploaded_files = st.file_uploader("📄 Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        combined_text = ""
        for file in uploaded_files:
            file_bytes = file.read()
            combined_text += extract_text_from_pdf(file_bytes)
        chunks = split_text(combined_text)
        vectors = get_embeddings(chunks)
        index = build_faiss_index(vectors)
    else:
        st.warning("Please upload at least one PDF to proceed.")
        st.stop()
else:
    elbert_index, elbert_chunks = load_elbert_data()
    if elbert_index is None:
        st.error("Precomputed index for Dr. Albert’s papers not found.")
        st.stop()
    chunks = elbert_chunks
    index = elbert_index

# --- Chat Interface ---
query = st.text_input("💬 Ask a question:")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if query:
    query_vec = model.encode([query])
    D, I = index.search(np.array([query_vec]), k=5)
    context = "\n\n".join([chunks[i] for i in I[0]])

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert assistant helping students understand research papers."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
    )

    answer = response.choices[0].message.content.strip()
    st.session_state.chat_history.append((query, answer))

# --- Display Chat History ---
if st.session_state.chat_history:
    st.write("## 🗂️ Chat History")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q{i+1}: {q}", expanded=False):
            st.markdown(f"**Answer:** {a}")

# --- Download Q&A Log ---
if st.session_state.chat_history:
    if st.button("📥 Download Q&A Log"):
        history_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
        st.download_button("📄 Download as .txt", data=history_text, file_name="chat_history.txt")
