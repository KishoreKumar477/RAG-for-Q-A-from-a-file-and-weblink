import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 RAG: Q&A from Files & Links")

# Helper Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar
with st.sidebar:
    st.header("Upload Sources")
    pdf_docs = st.file_uploader("PDFs", accept_multiple_files=True)
    url = st.text_input("Website URL")
    
    if st.button("Process"):
        with st.spinner("Processing..."):
            raw_text = ""
            if pdf_docs:
                raw_text += get_pdf_text(pdf_docs)
            if url:
                loader = WebBaseLoader(url)
                web_docs = loader.load()
                raw_text += " ".join([d.page_content for d in web_docs])
            
            if raw_text:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(raw_text)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
                st.success("Ready!")

# Main UI
if st.session_state.vectorstore:
    user_q = st.text_input("Ask a question:")
    if user_q:
        docs = st.session_state.vectorstore.similarity_search(user_q, k=3)
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": user_q})
        st.write(response["output_text"])
