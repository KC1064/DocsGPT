<h1 align="center" style="color:#00C49F;">DocGPT</h1>
<p align="center">
  <em><span style="color:#7B68EE;">Turn any URL into an intelligent, conversational document reader.</span></em>
</p>





---

## 🚀 Overview

A **Generative AI App** that lets you **paste any URL** and instantly builds a **chat interface** around the content of that page — making documents, articles, and blogs **interactive and easy to explore**.

---

## ⚙️ Tech Stack

| Tool          | Description                                 |
|---------------|---------------------------------------------|
| 🧱 LangChain   | For building LLM pipelines                  |
| 🧠 Gemini      | Google’s advanced LLM for smart responses   |
| 🗂️ ChromaDB    | Lightweight vector store for retrieval      |
| 🌐 BeautifulSoup | Scraping clean text from web pages       |
| 📟 Streamlit   | UI to chat and interact with the docs       |

---

## 📸 Features

- ✅ Input **any webpage URL**
- 🔍 Auto-scrapes & parses clean content
- 🧾 Stores chunks using **ChromaDB**
- 🤖 Ask context-aware questions with **Gemini**
- 🧑‍💻 Minimal and smooth **Streamlit UI**

---

## 🛠️ Setup in Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/genai-url-chat.git
cd genai-url-chat

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows

# 3. Install dependencies
pip install -r requirements.txt
````

> 🔐 **Set your Gemini API Key**

```bash
export GOOGLE_API_KEY=your_gemini_api_key_here  # Linux/Mac
set GOOGLE_API_KEY=your_gemini_api_key_here     # Windows
```

```bash
# 4. Run the app
streamlit run app.py
```

---

## 🧠 Behind the Scenes

```
[URL Input] 🔗
     |
[Scrape Clean Text] 🍜 using BeautifulSoup
     |
[Chunk & Embed] 🧩 with LangChain + ChromaDB
     |
[Query & Retrieve] 🧠 using Gemini
     |
[Streamlit Chat UI] 💬
```

---


