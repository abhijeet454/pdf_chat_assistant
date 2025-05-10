
# PDF Chat Assistant

PDF Chat Assistant is an interactive tool built with Streamlit, enabling users to upload PDF files and ask questions about the content. It utilizes Google Generative AI embeddings and FAISS for vector storage to provide accurate and detailed answers.

## Features

- Upload multiple PDF files.
- Extract text from PDFs and split it into manageable chunks.
- Generate embeddings for text chunks and store them in a FAISS vector store.
- Answer questions based on the content of the uploaded PDFs.
- Text-to-speech functionality to read out the answers.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abhijeet454/pdf_chat_assistant.git
   cd pdf-chat-assistant
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your Google API key:
   ```plaintext
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Open your browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Upload your PDF files using the sidebar menu.

4. Enter your question in the text input field and click "Find Answer".

5. The answer will be displayed, and the text-to-speech functionality will read it out loud.

## Project Structure

- `main.py`: The main application script.
- `requirements.txt`: List of required Python packages.
- `styles.css`: Custom CSS for styling the Streamlit app.
- `.env`: Environment file containing sensitive information (not included in the repository).

## Dependencies

- `streamlit`
- `PyPDF2`
- `langchain`
- `langchain_google_genai`
- `FAISS`
- `python-dotenv`
- `pyttsx3`

## Contact

For any inquiries or issues, please contact:

ABHIJEET KUMAR:(abhijeetk6744@gmail.com)

---

© 2024 ABHIJEET KUMAR. All rights reserved.
