import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Question Answerer", layout="wide")
st.title("📄 RAG: Q&A from Files & Links")

# --- Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def load_website(url):
    loader = WebBaseLoader(url)
    return loader.load()

def get_conversational_chain():
    # Looks for GOOGLE_API_KEY in Streamlit Secrets
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff")
    return chain

# --- Session State ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Sidebar ---
with st.sidebar:
    st.header("Data Sources")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    url = st.text_input("Or paste a Website URL")
    
    if st.button("Process Data"):
        all_docs = []
        with st.spinner("Processing..."):
            # 1. Get text from PDFs
            if pdf_docs:
                raw_text = get_pdf_text(pdf_docs)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(raw_text)
                all_docs.extend(chunks)
            
            # 2. Get text from URL
            if url:
                web_docs = load_website(url)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                web_chunks = text_splitter.split_documents(web_docs)
                all_docs.extend([d.page_content for d in web_chunks])

            if all_docs:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vectorstore = FAISS.from_texts(all_docs, embeddings)
                st.success("Database Ready!")

# --- Main UI ---
if st.session_state.vectorstore:
    user_question = st.text_input("Ask a question about your data:")
    if user_question:
        # Search the vector store
        docs = st.session_state.vectorstore.similarity_search(user_question, k=3)
        
        # Generate Answer
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": user_question})
        
        st.subheader("Answer:")
        st.write(response["output_text"])
else:
    st.info("Please upload a file or enter a URL and click 'Process Data' to start.")
