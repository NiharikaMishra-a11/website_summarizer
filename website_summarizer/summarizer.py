import os
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load API key from .env
load_dotenv()
api_key = os.getenv("SAMBANOVA_API_KEY") or os.getenv("OPENAI_API_KEY")

# Initialize client
client = OpenAI(api_key=api_key, base_url="https://api.sambanova.ai/v1")

# Get headless Chrome driver
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--remote-debugging-port=9222")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# Extract article content
def extract_article_text(url):
    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()

        paragraphs = soup.find_all('p')
        content = ' '.join(p.get_text() for p in paragraphs if len(p.get_text()) > 40)
        return content.strip()
    except Exception as e:
        return f"❌ Error while scraping: {str(e)}"

# Summarize using API
def summarize_text(text):
    try:
        if not text or text.startswith("❌"):
            return text

        response = client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",  # used model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes news articles."},
                {"role": "user", "content": f"Summarize the following article:\n\n{text}"}
            ],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error from AI: {str(e)}"

# Streamlit UI
st.title("📰 Website Summarizer")
url = st.text_input("Enter the article URL")

if st.button("Summarize"):
    with st.spinner("🔎 Extracting article..."):
        article_text = extract_article_text(url)

    with st.spinner("🤖 Generating summary..."):
        summary = summarize_text(article_text)

    st.subheader("✅ Summary:")
    st.write(summary)
