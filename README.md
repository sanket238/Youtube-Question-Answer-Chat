# 🎥 YouTube Video Q&A Streamlit App

This is a **Retrieval-Augmented Generation (RAG)** application built with **Streamlit** and **LangChain**.  
It enables users to ask questions about the content of any English-language YouTube video by fetching and processing the transcript, then using an LLM to provide context-aware answers.

---

## ✨ Features

- **Video Playback:** Watch the video directly within the Streamlit interface.  
- **Transcript Handling:** Fetches English transcripts via `youtube-transcript-api`. Provides clear error messages if transcripts are disabled or unavailable.  
- **Persistent Q&A:** Uses Streamlit's forms and session state for fast, responsive Q&A, avoiding unnecessary page reruns.  
- **RAG Pipeline:** Utilizes a LangChain workflow for data preparation and retrieval:
  - **Splitting:** Chunks the transcript (`RecursiveCharacterTextSplitter`).  
  - **Embedding/Store:** Embeds chunks using `all-MiniLM-L6-v2` and stores them in FAISS.  
  - **Retrieval:** Uses Maximum Marginal Relevance (MMR) for diverse context.  
  - **LLM Integration:** Answers questions using the `Mistral-7B-Instruct-v0.3` model hosted on the HuggingFace Inference Endpoint.  

---

## 🛠️ Technology Stack

- **Frontend/App Framework:** Streamlit  
- **Orchestration/RAG:** LangChain  
- **Transcript API:** youtube-transcript-api  
- **LLM:** Mistral-7B-Instruct-v0.3 (via HuggingFace Inference Endpoint)  
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`  
- **Vector Store:** FAISS  

---

## ⚙️ Setup and Installation

### 1. Prerequisites

- Python 3.9+ is required.  
- A HuggingFace account to access LLM endpoints.

### 2. Environment Variables

Create a `.env` file and set your HuggingFace API key:

```env
HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN_HERE"

### **3. Install Dependencies**

pip install -r requirements.txt
