# Updated summarizer.py with tab fix and Chrome binary check

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
import shutil

load_dotenv()
api_key = os.getenv("SAMBANOVA_API_KEY") 
client = OpenAI(api_key=api_key, base_url="https://api.sambanova.ai/v1")

# Setup Selenium driver

def get_driver():
    options = Options()
    options.binary_location = "/usr/bin/google-chrome"
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Extract article from URL
def extract_article(url):
    try:
        st.toast("🌐 Extracting article...")
        driver = get_driver()
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs if len(p.get_text()) > 40)
        driver.quit()
        return text.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Extract text from PDF
def extract_pdf_text(uploaded_file):
    try:
        st.toast("📄 Reading PDF...")
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        return f"❌ PDF extract error: {str(e)}"

# Split text into chunks
def split_text(text, chunk_size=1500, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.create_documents([text])

# Vector index
def create_vector_index(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

# Summarization
def summarize_chunks(text, detail_level):
    final_summary = ""
    chunks = split_text(text)

    if detail_level == "Detailed":
        max_tokens = 300
        prompt_type = "Summarize this in well-formed paragraphs. Do NOT use bullet points."
    else:
        max_tokens = 100
        prompt_type = "Summarize this in 3–5 short bullet points only. No intro phrases like 'here is a summary'."

    for doc in chunks:
        try:
            st.toast("🤖 Summarizing...")
            response = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a summarizer that follows instructions exactly."},
                    {"role": "user", "content": f"{prompt_type}\n\n{doc.page_content}"}
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            final_summary += response.choices[0].message.content.strip() + "\n\n"
        except Exception as e:
            final_summary += f"❌ Error: {str(e)}"
            break

    return final_summary.strip()

# Q&A mode
def ask_question(text, user_question):
    docs = split_text(text)
    vector_index = create_vector_index(docs)
    retriever = vector_index.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(llm=client, retriever=retriever)
    return chain.run(user_question)

# ----------------------- UI ---------------------------
st.set_page_config(page_title="AI Summarizer + QA", layout="wide")
st.title("🧠 Smart Summarizer + Ask Your Docs")

# Initialize session state
for key in ["input_mode", "task_mode", "summary_style", "text"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Input mode selection
input_mode = st.selectbox("📥 Select Input Type", ["Article URL", "Text", "PDF"], index=0)
st.session_state.input_mode = input_mode

# Task mode selection
task_mode = st.radio("Choose Action", ["📝 Full Summary", "💬 Ask a Question"], index=0)
st.session_state.task_mode = task_mode

# Content input
if input_mode == "Article URL":
    url = st.text_input("Paste article URL")
    if st.button("Load Article"):
        st.session_state.text = extract_article(url)

elif input_mode == "Text":
    input_text = st.text_area("Paste your text", height=300)
    if st.button("Load Text"):
        st.session_state.text = input_text

elif input_mode == "PDF":
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded and st.button("Load PDF"):
        st.session_state.text = extract_pdf_text(uploaded)

# Execute selected task
if st.session_state.text:
    if task_mode == "📝 Full Summary":
        summary_style = st.radio("Choose Summary Style", ["Brief", "Detailed"], index=0)
        st.session_state.summary_style = summary_style
        with st.spinner("📝 Generating summary..."):
            result = summarize_chunks(st.session_state.text, summary_style)
        st.subheader("📄 Summary:")
        st.write(result)

    elif task_mode == "💬 Ask a Question":
        question = st.text_input("What do you want to know?")
        if st.button("Ask") and question.strip():
            with st.spinner("🔍 Finding answer..."):
                answer = ask_question(st.session_state.text, question)
            st.subheader("💬 Answer:")
            st.write(answer)
