import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Helper Functions ---
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split the text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generate and store vector embeddings using ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="./chroma_db")
    vector_store.persist()
    st.success("**ChromaDB index saved successfully!** 🎉")
    print("Files saved successfully.")
    return vector_store

def get_conversational_chain():
    """Create a conversational chain for Q&A."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Process user input to find the answer to the question."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write(f"**Answer:** {response['output_text']}")

# --- Streamlit UI ---
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with PDF", page_icon="📚", layout="wide")
    st.markdown(
        """
        <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .subheader {
            font-size: 20px;
            color: #2196F3;
            text-align: center;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 12px;
            font-size: 18px;
            width: 100%;
        }
        .button:hover {
            background-color: #45a049;
        }
        .text-input {
            background-color: #f2f2f2;
            padding: 12px;
            border-radius: 8px;
            width: 100%;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="title">Chat with Your PDF Files 📖</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Upload your PDF and ask questions based on the content!</div>', unsafe_allow_html=True)

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.header("📥 Upload PDF Files")
        pdf_docs = st.file_uploader("Choose your PDF files", accept_multiple_files=True, label_visibility="collapsed")
        if st.button("Submit & Process", key="process_button"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file! 📄")
            else:
                with st.spinner("Processing your PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("🎉 PDF processing complete! Ask questions below.")

    # User input for asking questions
    user_question = st.text_input("❓ Ask a Question", key="user_question", placeholder="Type your question here...", label_visibility="collapsed")
    if user_question:
        with st.spinner("Fetching the answer..."):
            user_input(user_question)

    # Displaying guide text if PDFs are not uploaded or questions are not asked
    if not pdf_docs:
        st.write("👉 **Step 1:** Upload PDF files in the sidebar.")
    if not user_question:
        st.write("👉 **Step 2:** Type a question to ask based on the uploaded content.")

    # Footer section with links to your project (resume link, etc.)
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;">
            <p style="color:#9E9E9E;">Made with ❤️ by Vedant Parnaik</p>
            <a href="https://github.com/vedantparnaik" target="_blank">
                <button class="button">View My GitHub</button>
            </a>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
