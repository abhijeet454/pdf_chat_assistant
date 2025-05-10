import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pyttsx3
import logging
import threading
import hashlib
from typing import List, Optional
from datetime import datetime

# --- Set Page Configuration ---
st.set_page_config(page_title="Advanced PDF Chat Assistant", page_icon="📄🤖", layout="wide")

# --- Configuration and Initialization ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("🔴 GOOGLE_API_KEY not found. Please configure it in your environment.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Constants
VECTOR_STORE_PATH = "faiss_index_pdf_chat"
DEFAULT_TTS_RATE = 150
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000
EMBEDDING_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"

# --- Helper Functions ---

@st.cache_resource
def initialize_tts_engine() -> Optional[pyttsx3.Engine]:
    """Initialize and cache the TTS engine."""
    try:
        engine = pyttsx3.init()
        logging.info("TTS engine initialized successfully.")
        return engine
    except RuntimeError as e:
        logging.error(f"Failed to initialize TTS engine: {e}")
        st.error("🔴 TTS engine initialization failed.")
        return None

tts_engine = initialize_tts_engine()

def speak_text(text: str, rate: int) -> None:
    """Convert text to speech with specified rate in a separate daemon thread."""
    if tts_engine and text.strip():
        def tts_thread():
            try:
                tts_engine.setProperty('rate', rate)
                tts_engine.say(text)
                tts_engine.runAndWait()
            except RuntimeError as e:
                logging.error(f"TTS runtime error: {e}")

        # Create a daemon thread that terminates when the main program exits
        thread = threading.Thread(target=tts_thread, daemon=True)
        thread.start()

def extract_pdf_text(pdf_docs: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    logging.warning(f"No text extracted from page {page_num} of {pdf.name}")
        except Exception as e:
            logging.error(f"Failed to process PDF {pdf.name}: {e}")
            st.error(f"⚠️ Error with {pdf.name}: {str(e)}")
    if not text.strip():
        st.warning("⚠️ No text could be extracted from the uploaded PDFs.")
    return text

def split_text_into_chunks(text: str) -> List[str]:
    """Split text into chunks for vectorization."""
    if not text.strip():
        logging.warning("Empty text provided for chunking.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_text(text)
    logging.info(f"Text split into {len(chunks)} chunks.")
    return chunks

def create_vector_store(text_chunks: List[str], content_hash: str) -> Optional[FAISS]:
    """Create and return a FAISS vector store."""
    if not text_chunks:
        logging.warning("No text chunks provided for vector store creation.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        logging.info(f"Vector store created with hash: {content_hash}")
        return vector_store
    except Exception as e:
        logging.error(f"Vector store creation failed: {e}")
        st.error("🔴 Vector store creation failed.")
        return None

def get_qa_chain() -> load_qa_chain:
    """Initialize a question-answering chain with a custom prompt."""
    prompt_template = """
    You are a PDF Analyzer AI. Answer questions using *only* the provided PDF context.
    Provide detailed, accurate responses. If the answer isn't in the context, state:
    "The answer is not available in the provided PDF documents."

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3, streaming=True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def is_salutation(question: str) -> bool:
    """Check if the question is a simple salutation or greeting."""
    salutations = ["hello", "hi", "hey", "how are you", "good morning", "good afternoon", "good evening"]
    question_lower = question.lower().strip()
    words = question_lower.split()
    if len(words) < 5:
        for sal in salutations:
            if sal in question_lower:
                return True
    return False

# --- Session State Initialization ---
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = bool(tts_engine)
if "tts_rate" not in st.session_state:
    st.session_state.tts_rate = DEFAULT_TTS_RATE

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("📄 PDF Chat Assistant ✨")
    st.markdown("Upload PDFs and query their contents!")
    st.session_state.tts_enabled = st.toggle("Enable TTS", value=st.session_state.tts_enabled)
    if st.session_state.tts_enabled and not tts_engine:
        st.warning("⚠️ TTS unavailable. Disabled.")
        st.session_state.tts_enabled = False
    st.session_state.tts_rate = st.slider(
        "Speech Rate", 50, 300, st.session_state.tts_rate, 10,
        disabled=not st.session_state.tts_enabled
    )
    pdf_docs = st.file_uploader("Select PDFs", type="pdf", accept_multiple_files=True)
    if st.button("🔄 Reset"):
        st.session_state.conversation = []
        st.session_state.vector_store = None
        st.session_state.processed_hash = None
        st.rerun()
    if st.button("🗑️ Clear Chat"):
        st.session_state.conversation = []
        st.rerun()

# --- Main Interface ---
st.title("🤖 Advanced PDF Chat Assistant")
st.markdown("Query your PDFs with AI precision.")

# Display Chat History with Speak Buttons
for i, msg in enumerate(st.session_state.conversation):
    with st.chat_message(msg["role"]):
        st.markdown(f"**{msg['timestamp']}** {msg['content']}")
        if msg["role"] == "assistant" and st.session_state.tts_enabled:
            if st.button("🔊 Speak", key=f"speak_{i}"):
                speak_text(msg["content"], st.session_state.tts_rate)

# Process Uploaded PDFs
if pdf_docs:
    pdf_contents = [doc.read() for doc in pdf_docs]
    content_hash = hashlib.sha256(b"".join(pdf_contents)).hexdigest()
    if st.session_state.processed_hash != content_hash:
        with st.status("⏳ Processing PDFs...", expanded=True) as status:
            raw_text = extract_pdf_text(pdf_docs)
            if not raw_text:
                status.update(label="⚠️ No text extracted.", state="warning")
            else:
                text_chunks = split_text_into_chunks(raw_text)
                st.session_state.vector_store = create_vector_store(text_chunks, content_hash)
                if st.session_state.vector_store:
                    st.session_state.processed_hash = content_hash
                    status.update(label="✅ Processing complete!", state="complete")

# Handle User Input
if prompt := st.chat_input("Ask about your PDFs..."):
    if not pdf_docs:
        st.warning("☝️ Upload PDFs first.")
    elif not st.session_state.vector_store:
        st.error("🔴 Vector store not ready.")
    else:
        # Add user message to chat history with timestamp
        st.session_state.conversation.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        with st.chat_message("user"):
            st.markdown(f"**{st.session_state.conversation[-1]['timestamp']}** {prompt}")

        # Check if the question is a salutation
        if is_salutation(prompt):
            response = "Hello! How can I assist you with your PDFs today?"
            st.session_state.conversation.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            with st.chat_message("assistant"):
                st.markdown(f"**{st.session_state.conversation[-1]['timestamp']}** {response}")
                if st.session_state.tts_enabled:
                    speak_text(response, st.session_state.tts_rate)
        else:
            # Generate AI response based on PDFs
            with st.chat_message("assistant"):
                try:
                    with st.spinner("🧠 Thinking..."):
                        chain = get_qa_chain()
                        relevant_docs = st.session_state.vector_store.similarity_search(prompt, k=5)
                        if not relevant_docs:
                            response = "I couldn't find any relevant sections in your PDF(s) for that question."
                            st.warning(response)
                        else:
                            response_placeholder = st.empty()
                            full_response = ""
                            for chunk in chain.stream({"input_documents": relevant_docs, "question": prompt}):
                                if "output_text" in chunk:
                                    full_response += chunk["output_text"]
                                    response_placeholder.markdown(full_response + "▌")
                            response_placeholder.markdown(full_response)
                            response = full_response
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    if st.session_state.tts_enabled:
                        speak_text(response, st.session_state.tts_rate)
                except Exception as e:
                    logging.error(f"Error during response generation: {e}")
                    error_msg = "🔴 An error occurred while generating the response."
                    st.error(error_msg)
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })

if not pdf_docs:
    st.info("📤 Start by uploading PDFs in the sidebar!")