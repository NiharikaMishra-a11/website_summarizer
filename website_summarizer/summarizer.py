import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load API key
load_dotenv()
api_key = os.getenv("SAMBANOVA_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.sambanova.ai/v1")

# Setup Chrome Driver
def get_driver():
    options = Options()
    options.binary_location = "/usr/bin/chromium"  # For EC2, use "chromium"
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Extract article from URL
def extract_article(url):
    try:
        st.toast("🌐 Extracting article...")
        driver = get_driver()
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text() for p in paragraphs if len(p.get_text()) > 40)
    except Exception as e:
        return f"❌ Error: {e}"

# Extract PDF text
def extract_pdf_text(uploaded_file):
    try:
        st.toast("📄 Reading PDF...")
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        return f"❌ PDF extract error: {e}"

# Text splitter
def split_text(text, chunk_size=1500, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.create_documents([text])

# FAISS index for RAG
def create_vector_index(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

# Summarizer
def summarize_chunks(text, detail_level):
    chunks = split_text(text)
    summary = ""
    prompt_type = (
        "Summarize this in 3–5 very short bullet points. Do not include 'here is the summary':"
        if detail_level == "Brief"
        else "Summarize this in clear paragraphs without bullet points or headers:"
    )
    max_tokens = 100 if detail_level == "Brief" else 300

    for chunk in chunks:
        try:
            st.toast("🤖 Summarizing...")
            response = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful, concise summarizer."},
                    {"role": "user", "content": f"{prompt_type}\n\n{chunk.page_content}"}
                ],
                max_tokens=max_tokens,
                temperature=0.4,
            )
            summary += response.choices[0].message.content.strip() + "\n\n"
        except Exception as e:
            summary += f"❌ Error: {e}"
            break

    return summary.strip()

# RAG Question Answering
def ask_question(text, user_question):
    docs = split_text(text)
    vector_index = create_vector_index(docs)
    retriever = vector_index.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(llm=client, retriever=retriever)
    return chain.run(user_question)

# ----------------------- UI ---------------------------

st.set_page_config(page_title="AI Summarizer + RAG Q&A", layout="wide")
st.title("🧠 Universal AI Summarizer + Ask Your Docs")

# Session state defaults
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Article URL"
if "task_mode" not in st.session_state:
    st.session_state.task_mode = "📝 Full Summary"
if "summary_style" not in st.session_state:
    st.session_state.summary_style = "Brief"

# Select inputs
st.session_state.input_mode = st.selectbox("📥 Select Input Type", ["Article URL", "Text", "PDF"])
st.session_state.task_mode = st.radio("Choose Action", ["📝 Full Summary", "💬 Ask a Question"])

# Input handlers
text = ""
if st.session_state.input_mode == "Article URL":
    url = st.text_input("Paste article URL")
    if st.button("Load Article"):
        text = extract_article(url)

elif st.session_state.input_mode == "Text":
    text_input = st.text_area("Paste your text here", height=300)
    if st.button("Load Text"):
        text = text_input

elif st.session_state.input_mode == "PDF":
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded and st.button("Load PDF"):
        text = extract_pdf_text(uploaded)

# Task handler
if text:
    if st.session_state.task_mode == "📝 Full Summary":
        st.session_state.summary_style = st.radio("Choose Summary Style", ["Brief", "Detailed"])
        with st.spinner("✍️ Summarizing..."):
            summary = summarize_chunks(text, st.session_state.summary_style)
        st.subheader("📄 Summary:")
        st.write(summary)

    elif st.session_state.task_mode == "💬 Ask a Question":
        question = st.text_input("❓ Ask a question from this content")
        if st.button("Ask") and question.strip():
            with st.spinner("🔍 Searching answer..."):
                answer = ask_question(text, question)
            st.subheader("💬 Answer:")
            st.write(answer)
