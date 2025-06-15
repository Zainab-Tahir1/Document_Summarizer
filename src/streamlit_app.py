import streamlit as st
import os
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import login  

login(token="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")

# === Load Summarization & Embedding Models ===
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,  
    device_map= "cpu",
)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Streamlit UI Setup ===
st.set_page_config(page_title="Document Summarizer", layout="wide")
st.title("Document Summarizer")
st.markdown("Upload a PDF, TXT or Markdown file and get a summary using RAG-style retrieval with LLAMA.")

# === Upload File ===
uploaded_file = st.file_uploader("", type=["pdf", "txt", "md"])

# === Helper: Extract Text ===
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".txt") or file.name.endswith(".md"):
        return file.read().decode("utf-8")
    return ""

# === Helper: Chunk Text ===
def chunk_text(text, chunk_size=2048, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

# === RAG-style Summarization ===
def summarize_with_rag(chunks, query, top_k=3):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    query_embedding = embedder.encode([query], convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    _, indices = index.search(query_embedding, top_k)
    selected_chunks = [chunks[i] for i in indices[0]]

    combined_text = "\n\n".join(selected_chunks)
    prompt = (
        f"User query: {query}\n"
        f"Relevant text:\n{combined_text}\n\n"
        "Summarize this for the user."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant who summarizes the information for the user."},
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=300,
    )

    

    summary = outputs[0]["generated_text"][-1]
    summary = summary["content"]
    return summary, selected_chunks

# === Main App Logic ===
if uploaded_file:
    raw_text = extract_text(uploaded_file)
    if raw_text:
        chunks = chunk_text(raw_text)
        st.success(f"Document loaded and split into {len(chunks)} chunks.")
        query = st.text_input("Enter your query (or general summary request):", value="Summarize this document")
        top_k = 5  # Fixed number of chunks to retrieve

        if st.button("Generate Summary"):
            with st.spinner("Retrieving and summarizing..."):
                summary, contexts = summarize_with_rag(chunks, query, top_k=top_k)
                st.subheader("Final Summary")
                st.write(summary)
                for i, context in enumerate(contexts):
                    with st.expander(f"Show retrieved chunk {i+1}"):
                        st.write(context)
    else:
        st.error("Failed to extract any text from the uploaded file.")
