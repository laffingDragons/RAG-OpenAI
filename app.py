import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import tempfile
import openai
from langchain.schema import Document

# Function to load and split documents
def load_and_split_document(uploaded_file):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())  # Write the uploaded file to the temp file
            tmp_file_path = tmp_file.name  # Get the temporary file path

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Check if documents were loaded
        if not documents:
            raise ValueError("No documents processed. The file might be empty or corrupt.")
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents)

        # Debug: Check the structure of documents
        st.write(f"Documents: {split_documents[:3]}")  # Output the first 3 chunks for debugging
        
        # Ensure documents are in the correct format (i.e., list of Document objects)
        processed_documents = [Document(page_content=doc['content'], metadata=doc.get('metadata', {})) for doc in split_documents]
        
        return processed_documents
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

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
    if documents:
        st.write(f"Document split into {len(documents)} chunks.")

        # Create the retriever using the loaded documents
        try:
            retriever = RetrievalQA.from_documents(documents, OpenAI())
        except Exception as e:
            st.error(f"Error creating retriever: {str(e)}")

        # User query input
        query = st.text_input("Ask a question:")

        if query:
            # Retrieve the answer from the model
            try:
                result = retriever.run(query)
                st.write(f"Answer: {result}")
            except Exception as e:
                st.error(f"Error retrieving answer: {str(e)}")