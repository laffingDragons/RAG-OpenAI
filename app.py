import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Load API keys from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Function to load and split documents
def load_and_split_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Streamlit UI
st.title("RAG Model - Retrieval Augmented Generation")
st.write("Upload a PDF document and interact with the RAG model.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    st.write("Document uploaded, processing...")

    # Process the document
    documents = load_and_split_document(uploaded_file)
    st.write(f"Document split into {len(documents)} chunks.")

    # Create the retriever using the loaded documents
    retriever = RetrievalQA.from_documents(documents, OpenAI())

    # User query input
    query = st.text_input("Ask a question:")

    if query:
        # Retrieve the answer from the model
        result = retriever.run(query)
        st.write(f"Answer: {result}")