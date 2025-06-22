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
import tempfile
import requests

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.schema import Document

load_dotenv()
api_key = os.getenv("SAMBANOVA_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.sambanova.ai/v1")

# Setup Chrome for EC2 or local
def get_driver():
    options = Options()
    options.binary_location = "/usr/bin/google-chrome"
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Scrape URL using Selenium + BeautifulSoup
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

# Extract PDF text
def extract_pdf_text(uploaded_file):
    try:
        st.toast("📄 Reading PDF...")
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        return f"❌ PDF extract error: {str(e)}"

# Split large text into chunks
def split_text(text, chunk_size=1500, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.create_documents([text])

# Create vector store (in-memory FAISS)
def create_vector_index(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

# Summarize all chunks one by one
def summarize_chunks(text, detail_level):
    final_summary = ""
    chunks = split_text(text)
    max_tokens = 300 if detail_level == "Detailed" else 150
    prompt_type = "in detail" if detail_level == "Detailed" else "briefly"

    for doc in chunks:
        try:
            st.toast("🤖 Summarizing...")
            response = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "You summarize clearly and concisely."},
                    {"role": "user", "content": f"Summarize this {prompt_type}:\n{doc.page_content}"}
                ],
                max_tokens=max_tokens,
                temperature=0.5,
            )
            final_summary += response.choices[0].message.content.strip() + "\n\n"
        except Exception as e:
            final_summary += f"❌ Error: {str(e)}"
            break
    return final_summary.strip()

# LangChain-based QA over indexed docs
def ask_question(text, user_question):
    docs = split_text(text)
    vector_index = create_vector_index(docs)
    retriever = vector_index.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(llm=client, retriever=retriever)
    return chain.run(user_question)

# Streamlit UI
st.set_page_config(page_title="Universal AI Summarizer + QA", layout="wide")
st.title("🧠 Smart Summarizer + Chat over Documents")

input_mode = st.selectbox("📥 Input Type", ["Article URL", "Text", "PDF"])
task_mode = st.radio("Choose Action", ["📝 Full Summary", "💬 Ask a Question"])

text = ""

# Get input
if input_mode == "Article URL":
    url = st.text_input("Paste article URL")
    if st.button("Load Article"):
        text = extract_article(url)

elif input_mode == "Text":
    text_input = st.text_area("Paste your text", height=300)
    if st.button("Load Text"):
        text = text_input

elif input_mode == "PDF":
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded and st.button("Load PDF"):
        text = extract_pdf_text(uploaded)

# Perform task
if text:
    if task_mode == "📝 Full Summary":
        detail_level = st.radio("Summary Style", ["Brief", "Detailed"])
        with st.spinner("🔄 Summarizing..."):
            output = summarize_chunks(text, detail_level)
        st.subheader("📄 Summary:")
        st.write(output)

    elif task_mode == "💬 Ask a Question":
        question = st.text_input("What do you want to know?")
        if st.button("Ask"):
            with st.spinner("🔍 Searching..."):
                answer = ask_question(text, question)
            st.subheader("💬 Answer:")
            st.write(answer)
