# Document Summarizer

A document summarization application built on:

- **Retrieval-Augmented Generation (RAG)**
- **FAISS** for fast semantic retrieval
- **SentenceTransformer** embeddings for dense vector representation
- **LLaMA 3.2 Instruct 1B** for local summarization
- **Streamlit** for a clean, interactive interface


## Features

- Accepts documents in `.pdf`, `.txt`, and `.md` formats  
- Performs semantic chunk retrieval based on the user's query  
- Summarizes relevant content using an open-source local LLM  
- Designed to run fully on **CPU** — no GPU or paid API required  
- Lightweight, local, and production-ready


## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv env
```

### 2. Activate Environment 
For Windows:
```bash
env\Scripts\activate
```
For Linux/Mac:
```bash
source env/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

## Run the Streamlit App

Go to src folder:
```bash
streamlit run streamlit_app.py
```

Then go to:
```
http://localhost:8501
```

## Data 

Some samples documents are in data folder.

