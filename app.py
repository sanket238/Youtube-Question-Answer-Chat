import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter # FIX: Import itemgetter for dictionary key extraction

load_dotenv()

# --- Cached Resource Setup (Prevents rebuild on every rerun) ---

@st.cache_resource(show_spinner="Initializing LLM...")
def get_llm():
    """Initializes and caches the HuggingFace LLM endpoint."""
    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
        )
    )

@st.cache_data(show_spinner="Loading Transcript...")
def load_video_transcript(video_url):
    """Loads and caches the transcript for the given video URL."""
    try:
        loader = YoutubeLoader.from_youtube_url(video_url,add_video_info=False)
        # Load transcript
        docs = loader.load()

        if not docs:
            return "No transcript found."

        # Flatten to plain text
        transcript = docs[0].page_content.strip()
        return transcript

    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    
    except Exception as e:
        error_message = str(e)
        # Check for the specific error indicating missing English transcript
        if "No transcripts were found for any of the requested language codes: ['en']" in error_message:
            return "Transcript error: Only English transcripts are supported for Q&A currently, and an English transcript was not found for this video."
        
        # Log the error for debugging
        print(f"Error loading transcript: {e}")
        return f"An error occurred: {e}"
    

@st.cache_resource(show_spinner="Creating Vector Store and Retriever...")
def setup_rag_components(document):
    """Creates and caches the vector store and retriever from the document."""
    if not document or document.startswith("Transcripts are disabled") or document.startswith("An error occurred"):
        return None

    # Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([document])

    # Embeddings (Cached)
    embeddings= HuggingFaceEmbeddings(
        model_name= "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Vector Store and Retriever
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 4,"lambda_mult": 0.5}
    )
    return retriever

# --- Chain Definition Functions ---

def format_docs(retrived_docs):
    """Formats the retrieved documents into a single context string."""
    context_text = "\n\n".join(doc.page_content for doc in retrived_docs)
    return context_text

def get_rag_chain(retriever, llm):
    """
    Defines the complete RAG chain with the fix for the 'dict' object error.
    """
    # Define prompt template
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context: {context}
        Question: {question}
        """,
        input_variables = ['context', 'question']
    )
    
    # FIX: Use itemgetter('question') to extract the string query
    # from the input dictionary before passing it to the retriever.
    parallel_chain = RunnableParallel({
        # 1. Extract 'question' string, pass to retriever, then format docs.
        'context': itemgetter('question') | retriever | RunnableLambda(format_docs),
        # 2. Extract 'question' string to pass to the final prompt.
        'question': itemgetter('question')
    })
    
    # define parser
    parser = StrOutputParser()

    return parallel_chain | prompt | llm | parser

# --- Streamlit Application Layout ---

# Streamlit page configuration
st.set_page_config(page_title="YouTube Q&A App", page_icon=":robot_face:",layout="wide")

st.title("🎥 YouTube Video Q&A")
st.write("Paste a YouTube video link, watch it, and ask questions about its content!")

# User input for YouTube URL
video_url = st.text_input("🔗 Enter YouTube video URL")

# Initialize LLM (cached)
llm = get_llm()

if video_url:
    # 1. Load data (Cached)
    document = load_video_transcript(video_url)

    if document.startswith("Transcripts are disabled") or document.startswith("An error occurred"):
        st.error(document)
    else:
        # 2. Setup RAG components (Cached)
        retriever = setup_rag_components(document)
        
        # 3. Define the complete RAG chain
        main_chain = get_rag_chain(retriever, llm)

        # Layout split: left(video), right(chat)
        left_column, right_column = st.columns(2)
        
        with left_column:
            st.video(video_url)
            # Display source info for context
            st.markdown("---")
            st.subheader("Transcript Status")
            st.success("Transcript loaded successfully and RAG system is ready!")

        with right_column:
            st.subheader("💬 Ask Questions")
            user_question = st.text_input("❓ Enter your question about the video:")
            
            if user_question:
                with st.spinner("⏳ Getting answer..."):
                    # The main_chain.invoke call is correct.
                    answer = main_chain.invoke({"question": user_question})
                    st.markdown(f"**Answer:** {answer}")

elif not video_url:
    st.info("Please enter a YouTube URL to begin the Q&A process.")
