import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader

# ── Environment setup ────────────────────────────────────────────────────────
load_dotenv()  # works locally with a .env file
os.environ["USER_AGENT"] = "RAGApp/1.0"

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Q&A App", page_icon="🧠", layout="wide")
st.title("🧠 RAG: Q&A from Files & Web Links")
st.caption("Upload PDFs / TXT files or paste a URL, then ask questions about the content.")

# ── Cached resources (loaded once, reused across reruns) ─────────────────────
@st.cache_resource
def load_embeddings():
    """Load HuggingFace embedding model once and cache it."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    """Load Gemini LLM once and cache it."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
    )

# ── Session state init ────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Helper functions ──────────────────────────────────────────────────────────
def extract_text_from_files(uploaded_files) -> str:
    """Extract text from uploaded PDF and TXT files."""
    text = ""
    for file in uploaded_files:
        if file.name.lower().endswith(".txt"):
            try:
                text += file.read().decode("utf-8") + "\n"
            except UnicodeDecodeError:
                st.warning(f"Could not decode {file.name} as UTF-8. Skipping.")
        elif file.name.lower().endswith(".pdf"):
            try:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception as e:
                st.warning(f"Could not read {file.name}: {e}")
        else:
            st.warning(f"Unsupported file type: {file.name}. Only PDF and TXT are supported.")
    return text


def extract_text_from_url(url: str) -> str:
    """Scrape and return text content from a URL."""
    loader = WebBaseLoader(url)
    web_docs = loader.load()
    return " ".join(d.page_content for d in web_docs)


def build_vectorstore(raw_text: str):
    """Chunk text and build a FAISS vectorstore."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_text(raw_text)
    embeddings = load_embeddings()
    return FAISS.from_texts(chunks, embeddings)


def get_answer(docs, question: str) -> str:
    try:
        """Run the RAG chain and return the LLM answer."""
        llm = load_llm()
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer the question based ONLY on the \
    context provided below. If the answer cannot be found in the context, respond \
    with: "I don't have enough information in the provided sources to answer this."
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:"""
        )
        chain = create_stuff_documents_chain(llm, prompt)
        return chain.invoke({"input": question, "context": docs})
    except Exception as e:
        if "ResourceExhausted" in str(type(e).__name__):
            return "⚠️ API rate limit reached. Please wait 60 seconds and try again."
        return f"⚠️ An error occurred: {str(e)}"

# ── Sidebar: source ingestion ─────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Add Your Sources")

    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        accept_multiple_files=True,
        type=["pdf", "txt"],
    )
    url = st.text_input("Or paste a Website URL", placeholder="https://example.com")

    col1, col2 = st.columns(2)

    with col1:
        process_clicked = st.button("⚙️ Process", use_container_width=True)

    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.rerun()

    if process_clicked:
        raw_text = ""

        with st.spinner("Reading sources..."):
            if uploaded_files:
                raw_text += extract_text_from_files(uploaded_files)

            if url:
                try:
                    raw_text += extract_text_from_url(url)
                except Exception as e:
                    st.warning(f"Could not load URL: {e}")

        if raw_text.strip():
            with st.spinner("Building vector index..."):
                st.session_state.vectorstore = build_vectorstore(raw_text)
                st.session_state.chat_history = []  # reset chat on new source
            st.success(f"✅ Ready! Indexed {len(raw_text):,} characters.")
        else:
            st.warning("No content found. Please upload a file or enter a valid URL.")

    # Status indicator
    st.divider()
    if st.session_state.vectorstore:
        st.success("🟢 Vector store is active")
    else:
        st.info("⚪ No sources loaded yet")

# ── Main area: chat interface ─────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.info("👈 Upload a file or enter a URL in the sidebar, then click **Process** to get started.")
    st.stop()

# Display chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        with st.expander("📚 Source chunks used"):
            for i, chunk in enumerate(entry["source_chunks"]):
                st.markdown(f"**Chunk {i + 1}:**")
                st.caption(chunk[:400] + ("..." if len(chunk) > 400 else ""))

# Chat input
user_question = st.chat_input("Ask a question about your documents...")

if user_question and user_question.strip():
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved_docs = st.session_state.vectorstore.similarity_search(
                user_question, k=4
            )
            answer = get_answer(retrieved_docs, user_question)

        st.write(answer)

        with st.expander("📚 Source chunks used"):
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"**Chunk {i + 1}:**")
                st.caption(
                    doc.page_content[:400]
                    + ("..." if len(doc.page_content) > 400 else "")
                )

    # Save to history
    st.session_state.chat_history.append({
        "question": user_question,
        "answer": answer,
        "source_chunks": [doc.page_content for doc in retrieved_docs],
    })
