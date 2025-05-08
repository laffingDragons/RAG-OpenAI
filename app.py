import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import tempfile
import openai
from langchain.schema import Document
import base64

# Set page configuration
st.set_page_config(
    page_title="Enhanced RAG Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the sidebar
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    .stSlider > div > div > div {
        background-color: #f63366;
    }
    .stCheckbox > label {
        color: white;
    }
    .sidebar-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'file_details' not in st.session_state:
    st.session_state.file_details = {}

# Function to load and split documents
def load_and_split_document(uploaded_files, chunk_size, chunk_overlap):
    try:
        all_documents = []
        
        for uploaded_file in uploaded_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())  # Write the uploaded file to the temp file
                tmp_file_path = tmp_file.name  # Get the temporary file path

            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Store file details
            st.session_state.file_details[uploaded_file.name] = {
                'path': tmp_file_path,
                'size': f"{uploaded_file.size/1000000:.1f}MB",
                'pages': len(documents)
            }

            # Add to all documents
            all_documents.extend(documents)
            
            # Clean up the temp file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        # Check if documents were loaded
        if not all_documents:
            raise ValueError("No documents processed. The files might be empty or corrupt.")
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        split_documents = text_splitter.split_documents(all_documents)

        return split_documents
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None

# Function to verify OpenAI API key
def verify_openai_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except Exception as e:
        st.error(f"API verification error: {str(e)}")
        return False

# Function to display PDF preview
def display_pdf_preview(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        pdf_display = f"""
            <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF preview: {str(e)}")

# SIDEBAR
st.sidebar.markdown("<div class='sidebar-header'>Settings</div>", unsafe_allow_html=True)

# Chunk Size Setting
st.sidebar.subheader("Chunk Size")
chunk_size = st.sidebar.slider("", min_value=500, max_value=2000, value=1000, step=100)

# Chunk Overlap Setting
st.sidebar.subheader("Chunk Overlap")
chunk_overlap = st.sidebar.slider("", min_value=0, max_value=500, value=200, step=50)

# Max PDFs Setting
st.sidebar.subheader("Max PDFs to Upload")
max_pdfs = st.sidebar.number_input("", min_value=1, max_value=10, value=3, step=1)

# PDF Preview Option
show_pdf_preview = st.sidebar.checkbox("Show PDF Preview", value=True)

# MAIN CONTENT
st.title("Enhanced RAG Model - Retrieval Augmented Generation")
st.write("Upload PDF(s), customize settings, and ask questions about the content.")

# Two column layout for API key and file upload
col1, col2 = st.columns([1, 1])

with col1:
    # Input field for OpenAI API Key
    api_key_input = st.text_input("Enter your OpenAI API Key:", type="password")

    # Verify the API key
    if api_key_input:
        if verify_openai_api_key(api_key_input):
            st.success("API Key is valid!")
            os.environ["OPENAI_API_KEY"] = api_key_input
        else:
            st.error("Invalid API Key. Please check and try again.")

# File Uploader
st.subheader("Upload PDF(s)")
uploaded_files = st.file_uploader("", 
                                  type="pdf", 
                                  accept_multiple_files=True,
                                  help=f"Maximum {max_pdfs} files, 200MB per file")

# Process uploaded files
if uploaded_files:
    # Check if number of files exceeds maximum
    if len(uploaded_files) > max_pdfs:
        st.warning(f"Maximum {max_pdfs} files allowed. Only the first {max_pdfs} will be processed.")
        uploaded_files = uploaded_files[:max_pdfs]
    
    # Display uploaded files
    for file in uploaded_files:
        file_size_mb = file.size / (1024 * 1024)
        st.info(f"{file.name} ({file_size_mb:.2f}MB)")
        
        # Store files in session state
        if file not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(file)
    
    # Process button
    if st.button("Process Documents"):
        if "OPENAI_API_KEY" in os.environ:
            with st.spinner("Processing documents..."):
                # Process the documents
                documents = load_and_split_document(uploaded_files, chunk_size, chunk_overlap)
                
                if documents and len(documents) > 0:
                    st.session_state.processed_documents = documents
                    st.success(f"Documents processed! {len(documents)} chunks created.")
                    
                    # Create embeddings and vector store
                    try:
                        with st.spinner("Creating embeddings and vector store..."):
                            embeddings = OpenAIEmbeddings()
                            vectorstore = FAISS.from_documents(documents, embeddings)
                            retriever = vectorstore.as_retriever()
                            
                            # Create the QA chain
                            llm = OpenAI(temperature=0)
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type="stuff",
                                retriever=retriever
                            )
                            
                            st.session_state.qa_chain = qa_chain
                            st.success("RAG model ready! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Error creating retriever: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                elif documents is not None:
                    st.error("No document chunks were created. The PDFs might be empty or unreadable.")
        else:
            st.warning("Please enter a valid OpenAI API key before processing documents.")

# Display PDF preview if enabled
if show_pdf_preview and len(st.session_state.uploaded_files) > 0:
    st.subheader("PDF Preview")
    
    # Create an expander for the preview
    with st.expander("PDF Preview", expanded=True):
        # Check if we have file details with paths
        if st.session_state.file_details:
            # Select which file to preview
            file_names = list(st.session_state.file_details.keys())
            selected_file = st.selectbox("Select file to preview", file_names)
            
            if selected_file and selected_file in st.session_state.file_details:
                file_path = st.session_state.file_details[selected_file]['path']
                if os.path.exists(file_path):
                    display_pdf_preview(file_path)
                else:
                    st.warning("File preview not available. Please reupload the file.")
        else:
            st.info("Upload and process files to see preview")

# Question answering interface (only show if we have a QA chain)
if 'qa_chain' in st.session_state:
    st.subheader("Ask Questions")
    
    query = st.text_input("Enter your question:")
    
    if query:
        try:
            with st.spinner("Generating answer..."):
                result = st.session_state.qa_chain.run(query)
                
                # Display answer in a nice format
                st.markdown("### Answer")
                st.write(result)
        except Exception as e:
            st.error(f"Error retrieving answer: {str(e)}")