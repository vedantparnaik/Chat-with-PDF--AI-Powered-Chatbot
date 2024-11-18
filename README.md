# Chat with PDF - AI-Powered Chatbot

**Chat with PDF** is an AI-powered chatbot that allows users to upload PDF documents and interact with them by asking questions based on the content of the PDF. Using **Google's Gemini Model** for natural language understanding, this project extracts information from uploaded PDFs and provides detailed answers, making it an ideal tool for knowledge retrieval, document analysis, and research purposes.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python-based Web Framework)
- **Backend**: Python, LangChain, Google Gemini
- **Embeddings**: Google Generative AI Embeddings
- **Text Vectorization**: Chroma
- **PDF Parsing**: PyPDF2
- **Environment Variables**: `.env` (for storing API keys)

## 💡 Features

- **PDF Upload**: Upload multiple PDF files through an intuitive UI.
- **Question Answering**: Ask questions about the content of the uploaded PDFs.
- **AI Integration**: Built with **Google Generative AI** to generate accurate, context-driven responses.
- **Text Splitting**: Large PDFs are split into manageable chunks for more efficient processing.
- **Persistent Vector Storage**: Use **ChromaDB** for storing document embeddings, enabling fast similarity searches.

## 🚀 Getting Started

### Prerequisites

1. **Python 3.x**: Make sure you have Python 3.7+ installed.
2. **Required Libraries**: Install dependencies from `requirements.txt`.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/chat-with-pdf.git
    cd chat-with-pdf
    ```

2. Create and activate a **virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # For macOS/Linux
    .\venv\Scripts\activate  # For Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up your **Google API key**:
    - Create a `.env` file in the project root directory and add your Google API key:
    ```bash
    GOOGLE_API_KEY=your_google_api_key_here
    ```

### Run the Application

1. Launch the app:
    ```bash
    streamlit run app.py
    ```

2. Open the application in your browser:
    - The app will be available at `http://localhost:8501` by default.

### Usage

1. **Upload PDF Files**: Use the file uploader on the sidebar to upload your PDFs.
2. **Process PDFs**: Click on the "Submit & Process" button to extract the text from the PDFs and create vector embeddings.
3. **Ask Questions**: Enter your questions in the input box, and the chatbot will provide answers based on the uploaded PDFs.

### Example Questions

- "What is the main topic of the document?"
- "What are the key findings in the report?"
- "Can you summarize the document?"

---
