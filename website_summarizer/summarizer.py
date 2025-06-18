import os
from dotenv import load_dotenv
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import streamlit as st

# ✅ Load API key from .env
load_dotenv()
api_key = os.getenv("SAMBANOVA_API_KEY")

# 🔐 Initialize SambaNova OpenAI client with secure key
client = OpenAI(
    api_key=api_key,
    base_url="https://api.sambanova.ai/v1"
)


# ✅ Extract article using Selenium + BeautifulSoup
def extract_article_with_selenium(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

        service = Service(ChromeDriverManager().install())  # ✅ New way to manage driver
        driver = webdriver.Chrome(service=service, options=chrome_options)

        driver.get(url)
        time.sleep(5)

        html = driver.page_source
        driver.quit()

        soup = BeautifulSoup(html, 'html.parser')
        paragraphs = soup.find_all('p')

        text = "\n".join(
            p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40
        )

        print("\n🔍 Extracted Text Preview:\n", text[:1000])
        return text.strip()
    except Exception as e:
        return f"Error extracting article: {str(e)}"

# ✅ Use working SambaNova model
def summarize_with_sambanova(text):
    try:
        response = client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",  # ✅ Model your key supports
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes long articles clearly."},
                {"role": "user", "content": f"Summarize this article:\n\n{text}"}
            ],
            temperature=0.5,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from AI: {str(e)}"

# ✅ Streamlit App
st.set_page_config(page_title="SambaNova Article Summarizer")
st.title("🧠 Blog & Article Summarizer (SambaNova + Selenium)")
st.markdown("Paste a Medium/blog/news article URL below:")

url = st.text_input("🔗 Enter article URL")

if st.button("Summarize"):
    if not url:
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("📖 Extracting content..."):
            article = extract_article_with_selenium(url)
            if not article:
                st.error("❌ Couldn’t extract enough article content.")
            else:
                with st.spinner("✍️ Summarizing with SambaNova..."):
                    summary = summarize_with_sambanova(article)
                    st.success("✅ Summary:")
                    st.write(summary)
