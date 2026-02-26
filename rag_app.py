import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from pypdf import PdfReader

st.set_page_config(page_title="Simple RAG", layout="wide")
st.title("Simple RAG - Document & Website Q&A")

# --- Session state init ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "source_name" not in st.session_state:
    st.session_state.source_name = None

# --- Helpers ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")

def load_website(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

def build_vectorstore_from_text(text, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings), len(chunks)

def build_vectorstore_from_docs(documents, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings), len(chunks)

# --- Sidebar ---
with st.sidebar:
    st.header("Choose Data Source")

    option = st.radio(
        "Select Source Type",
        ["Upload Document (PDF/TXT)", "Paste Website URL"]
    )

    # ---- FILE UPLOAD ----
    if option == "Upload Document (PDF/TXT)":
        uploaded_file = st.file_uploader(
            "Drag & Drop or Browse File",
            type=["pdf", "txt"]
        )

        if uploaded_file:
            if uploaded_file.name != st.session_state.source_name:
                with st.spinner("Processing document..."):
                    embeddings = load_embeddings()
                    text = extract_text(uploaded_file)
                    vs, n_chunks = build_vectorstore_from_text(text, embeddings)

                    st.session_state.vectorstore = vs
                    st.session_state.source_name = uploaded_file.name

                st.success(f"Stored {n_chunks} chunks from {uploaded_file.name}")
            else:
                st.info(f"Loaded: {st.session_state.source_name}")

    # ---- WEBSITE URL ----
    if option == "Paste Website URL":
        url = st.text_input("Enter Website URL")

        if url:
            if url != st.session_state.source_name:
                with st.spinner("Fetching and processing website..."):
                    embeddings = load_embeddings()
                    documents = load_website(url)
                    vs, n_chunks = build_vectorstore_from_docs(documents, embeddings)

                    st.session_state.vectorstore = vs
                    st.session_state.source_name = url

                st.success(f"Stored {n_chunks} chunks from website")
            else:
                st.info(f"Loaded: {st.session_state.source_name}")

    if st.session_state.vectorstore and st.button("Clear Data"):
        st.session_state.vectorstore = None
        st.session_state.source_name = None
        st.rerun()

# --- Main: Q&A ---
if st.session_state.vectorstore is None:
    st.info("Upload a document or paste a website URL to begin.")
else:
    query = st.text_input(
        "Ask a question",
        placeholder="e.g. What is this content about?"
    )

    if query:
        with st.spinner("Searching..."):
            docs = st.session_state.vectorstore.similarity_search(query, k=3)

        st.subheader("Retrieved Context")
        for i, doc in enumerate(docs, 1):
            with st.expander(f"Chunk {i}", expanded=(i == 1)):
                st.write(doc.page_content)