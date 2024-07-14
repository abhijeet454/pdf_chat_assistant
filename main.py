import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path
import pyttsx3
import logging

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from the environment
google_api_key = os.getenv("GEMINI_API_KEY")
if google_api_key is None:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in the .env file.")

# Set up Google API key in environment
os.environ["GOOGLE_API_KEY"] = google_api_key

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("index1.faiss")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the questions as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the given file." Provide accurate answers.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and return answers
def user_input(pdf_docs, user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not Path("index1.faiss").is_file():
        text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(text)
        get_vector_store(text_chunks)

    new_db = FAISS.load_local("index1.faiss", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        logging.error(f"Error during API call: {e}")
        return "An error occurred while processing your request."

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="PDF Chat Assistant", page_icon=":robot_face:")
    st.title("PDF Chat Assistant :robot_face:")
    st.markdown("**Interact with your PDF documents**")

    # Load custom CSS
    with open("styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.sidebar.header("Menu:")
    st.sidebar.write("1. Upload your PDF files :file_folder:")

    pdf_docs = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if pdf_docs:
        with st.spinner("Processing your PDF documents..."):
            text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(text)
            get_vector_store(text_chunks)
        st.success("PDF documents processed successfully!")

    user_question = st.text_input("Ask a question about your PDF documents:")

    if st.button("Find Answer"):
        if user_question:
            if pdf_docs:
                with st.spinner("Searching for the answer..."):
                    reply = user_input(pdf_docs, user_question)
                    st.write("**Answer:** ", reply)
                    # Use pyttsx3 to read out the answer
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(reply)
                    engine.runAndWait()
            else:
                st.warning("Please upload PDF files first.")
        else:
            st.warning("Please enter a question.")

if __name__ == '__main__':
    main()
