import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import tempfile
import openai

# Function to load and split documents
def load_and_split_document(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())  # Write the uploaded file to the temp file
        tmp_file_path = tmp_file.name  # Get the temporary file path

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Function to verify OpenAI API key
def verify_openai_api_key(api_key):
    try:
        # Set the OpenAI API key
        openai.api_key = api_key
        # Make a simple API call to check the validity
        openai.Model.list()  # This will list available models
        return True
    except Exception as e:
        return False

# Streamlit UI
st.title("RAG Model - Retrieval Augmented Generation")
st.write("Upload a PDF document and interact with the RAG model.")

# Input field for OpenAI API Key
api_key_input = st.text_input("Enter your OpenAI API Key:", type="password")

# Verify the API key
if api_key_input:
    if verify_openai_api_key(api_key_input):
        st.success("API Key is valid!")
        # Store the API key for use in the app
        os.environ["OPENAI_API_KEY"] = api_key_input
    else:
        st.error("Invalid API Key. Please check and try again.")

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