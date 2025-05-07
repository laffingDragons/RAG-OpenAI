import streamlit as st
import os
import tempfile
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import fitz  # PyMuPDF for PDF preview
from datetime import datetime
import pandas as pd

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Function to load and split documents
@st.cache_data
def load_and_split_documents(uploaded_files, chunk_size=1000, chunk_overlap=200):
    documents = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            documents.extend(docs)
            os.unlink(tmp_file_path)  # Clean up
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

# Function to create PDF preview
def get_pdf_preview(uploaded_file, max_pages=2):
    try:
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        preview = ""
        for page_num in range(min(max_pages, pdf_document.page_count)):
            page = pdf_document.load_page(page_num)
            preview += page.get_text("text")[:500] + "\n\n"
        pdf_document.close()
        return preview
    except Exception as e:
        return f"Error generating preview: {str(e)}"

# Initialize embeddings and model
@st.cache_resource
def initialize_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = HuggingFacePipeline.from_model_id(
        model_id="distilgpt2",
        task="text-generation",
        pipeline_kwargs={"max_length": 200, "num_return_sequences": 1},
    )
    return embeddings, llm

# Streamlit UI
st.title("Enhanced RAG Model - Retrieval Augmented Generation")
st.write("Upload PDF(s), customize settings, and ask questions about the content.")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, step=50)
    max_files = st.number_input("Max PDFs to Upload", 1, 10, 3)
    show_preview = st.checkbox("Show PDF Preview", value=True)

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF(s)", type="pdf", accept_multiple_files=True, key="pdf_uploader"
)

if uploaded_files:
    if len(uploaded_files) > max_files:
        st.warning(f"Too many files! Only processing the first {max_files} PDFs.")
        uploaded_files = uploaded_files[:max_files]

    # Show PDF preview
    if show_preview:
        with st.expander("PDF Preview"):
            for uploaded_file in uploaded_files:
                st.subheader(uploaded_file.name)
                preview = get_pdf_preview(uploaded_file)
                st.text_area("Preview", preview, height=150)

    # Process documents
    with st.spinner("Processing documents..."):
        documents = load_and_split_documents(
            uploaded_files, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not documents:
            st.error("No documents processed. Please check the files and try again.")
        else:
            st.success(f"Processed {len(documents)} document chunks.")

            # Create vector store and retriever
            embeddings, llm = initialize_models()
            vector_store = FAISS.from_documents(documents, embeddings)
            st.session_state.retriever = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
            )
            st.session_state.processed = True

# Query input
if st.session_state.processed:
    query = st.text_input("Ask a question about the document(s):")
    if query:
        with st.spinner("Generating answer..."):
            try:
                result = st.session_state.retriever({"query": query})["result"]
                st.write(f"**Answer**: {result}")

                # Save to query history
                st.session_state.query_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "answer": result
                })
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

# Display query history
if st.session_state.query_history:
    with st.expander("Query History"):
        history_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(history_df)

        # Export history
        if st.button("Export History"):
            history_df.to_csv("query_history.csv", index=False)
            st.success("History exported as query_history.csv")

# Reset button
if st.button("Reset"):
    st.session_state.clear()
    st.experimental_rerun()