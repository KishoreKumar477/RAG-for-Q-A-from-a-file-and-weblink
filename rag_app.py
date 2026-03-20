import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()
os.environ["USER_AGENT"] = "RAGApp/1.0"

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 RAG: Q&A from Files & Links")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_answer(docs, question):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.

Context: {context}

Question: {input}
""")
    chain = create_stuff_documents_chain(model, prompt)
    return chain.invoke({"input": question, "context": docs})

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

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
                try:
                    loader = WebBaseLoader(url)
                    web_docs = loader.load()
                    raw_text += " ".join([d.page_content for d in web_docs])
                except Exception as e:
                    st.warning(f"Could not load URL: {e}")
            
            if raw_text:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(raw_text)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
                st.success("Ready!")

if st.session_state.vectorstore:
    user_q = st.text_input("Ask a question:")
    if user_q:
        docs = st.session_state.vectorstore.similarity_search(user_q, k=3)
        response = get_answer(docs, user_q)
        st.write(response)
