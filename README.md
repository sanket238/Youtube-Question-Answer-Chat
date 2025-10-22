🎥 YouTube Video Q&A Streamlit App

This project is a sophisticated Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain, designed to allow users to ask questions about the content of any publicly available English-language YouTube video.

The application fetches the video transcript, processes it, and uses a powerful Large Language Model (LLM) to provide context-aware answers.

✨ Features

Video Playback: Watch the YouTube video directly within the Streamlit interface.

Transcript Loading: Automatically fetches the English transcript using youtube-transcript-api.

Error Handling: Provides clear, concise error messages if transcripts are disabled or unavailable in English.

Persistent Q&A: Uses Streamlit's st.form and st.session_state to prevent unnecessary script reruns, ensuring a fast and responsive chat experience.

RAG Pipeline: Utilizes LangChain for a complete RAG workflow:

Splitting: Breaks the transcript into manageable chunks (RecursiveCharacterTextSplitter).

Embedding: Embeds chunks using a local HuggingFace model (all-MiniLM-L6-v2).

Vector Store: Stores embeddings in a FAISS vector store.

Retrieval: Uses Maximum Marginal Relevance (MMR) search for diverse context retrieval.

LLM Integration: Answers questions using the Mistral-7B-Instruct-v0.3 model hosted on the HuggingFace Inference Endpoint.

🛠️ Technology Stack

Frontend/App Framework: Streamlit

Orchestration/RAG: LangChain

Transcript API: youtube-transcript-api

LLM: Mistral-7B-Instruct-v0.3 (via HuggingFace Inference Endpoint)

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector Store: FAISS

⚙️ Setup and Installation

1. Prerequisites

You need a Python environment (3.9+) to run this application.

2. Environment Variables

Create a file named .env in the project root directory and add your HuggingFace API key:

HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN_HERE"


3. Install Dependencies

Install all necessary Python packages:

pip install streamlit youtube-transcript-api langchain_text_splitters langchain_huggingface langchain_community python-dotenv


4. Run the Application

Save the provided code as youtube_qa_app.py and run it using Streamlit:

streamlit run youtube_qa_app.py


The application will open in your web browser (usually at http://localhost:8501).

🚀 Usage

Enter URL: Paste the full URL of a YouTube video into the text input field.

Wait for Setup: The application will display spinners as it loads the transcript and sets up the RAG components (this is only done once per URL due to caching).

Ask Questions: Use the "Ask Questions" section. Type your question and click "Get Answer" to receive a context-specific response from the video transcript.
