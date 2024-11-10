import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.vectorstores import Chroma

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="./chroma_db")
    # Save the vector store
    vector_store.persist()
    st.success("Saved successfully!")
    print("Files saved successfully.")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    # new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    # print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📚")
    st.header("Chat with Your PDF Files")
    st.write(
        """
        Upload your PDF files, and ask questions about their content. This AI-powered chatbot 
        will read through the documents and provide answers based on the content.
        """
    )
    # user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.header("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # User input for asking questions
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        with st.spinner("Finding answer..."):
            user_input(user_question)

    # Display example for better user experience
    if not pdf_docs:
        st.write("👉 Upload PDFs to get started.")
    if not user_question:
        st.write("👉 Type a question above to ask the chatbot.")


if __name__ == "__main__":
    main()
