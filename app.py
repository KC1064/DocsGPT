# import os
# import uuid
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Chroma

# import chromadb
# import google.generativeai as genai
# import streamlit as st
# import json

# # Load API Key
# load_dotenv()
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# # Configure Gemini API
# genai.configure(api_key=GOOGLE_API_KEY)

# # Initialize Google Generative AI model
# @st.cache_resource
# def get_model():
#     return genai.GenerativeModel('gemini-1.5-flash')

# # Setup Embedding Model
# @st.cache_resource
# def get_embedding_model():
#     return GoogleGenerativeAIEmbeddings(
#         model="models/text-embedding-004",
#         google_api_key=GOOGLE_API_KEY
#     )

# # Setup ChromaDB
# @st.cache_resource
# def get_chroma_client():
#     chroma_client = chromadb.PersistentClient(path="./chroma_db")
#     collection_name = "website_crawling"
#     embedding_model = get_embedding_model()
    
#     return Chroma(
#         client=chroma_client,
#         collection_name=collection_name,
#         embedding_function=embedding_model,
#     )

# def get_html_content(url, title='firstPage'):
#     folder_path = f'./{title}'
#     os.makedirs(folder_path, exist_ok=True)
#     response = requests.get(url)
#     with open(os.path.join(folder_path, 'output.html'), 'w', encoding='utf-8') as f:
#         f.write(response.text)
#     return response.text

# def parse_html(html, base_url):
#     soup = BeautifulSoup(html, 'html.parser')
#     page_title = soup.title.string if soup.title else 'No Title'
#     page_content = soup.get_text(separator='\n', strip=True)
    
#     links = []
#     for a in soup.find_all('a', href=True):
#         text = a.get_text(strip=True)
#         href = a['href']
#         full_url = urljoin(base_url, href)
#         if full_url.startswith("http"):
#             links.append({'text': text, 'url': full_url})
    
#     return {
#         'page_title': page_title,
#         'page_content': page_content,
#         'urls': links
#     }

# # ----------- 🧠 Crawling and Storing Embeddings -----------
# def crawl_and_store(url, progress_bar=None, status_text=None):
#     chroma_langchain = get_chroma_client()
    
#     if status_text:
#         status_text.text("Fetching main page...")
    
#     html = get_html_content(url)
#     website_data = parse_html(html, base_url=url)
#     combined_content = website_data['page_content']
    
#     total_links = len(website_data['urls'])
    
#     for i, link in enumerate(website_data['urls']):
#         try:
#             if status_text:
#                 status_text.text(f"Crawling page {i+1}/{total_links}: {link['text'][:50]}...")
            
#             sub_html = get_html_content(link['url'], title=f'page_{i}')
#             sub_data = parse_html(sub_html, base_url=link['url'])
#             combined_content += '\n' + sub_data['page_content']
            
#             if progress_bar:
#                 progress_bar.progress((i + 1) / total_links)
                
#         except Exception as e:
#             if status_text:
#                 status_text.text(f"Skipping {link['url']}: {str(e)[:50]}...")
#             continue

#     if status_text:
#         status_text.text("Processing and storing content...")

#     # Split content
#     splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
#     chunks = splitter.split_text(combined_content)

#     # Store chunks in Chroma
#     ids = [str(uuid.uuid4()) for _ in chunks]
#     chroma_langchain.add_texts(texts=chunks, ids=ids)
    
#     return len(chunks)

# # ----------- 🤖 QA Bot with Gemini + ChromaDB -----------
# def query_bot(user_query):
#     model = get_model()
#     chroma_langchain = get_chroma_client()
    
#     system_prompt = '''You are a helpful AI Assistant. 
# Your task is to answer user queries using only the context provided from ChromaDB.
# If the context does not contain relevant information, respond with:
# "Chroma DB operation failed - no relevant information found."

# Respond in this format:
# "ai_response": "answer using context here",

# }
# '''

#     try:
#         results = chroma_langchain.similarity_search(user_query, k=3)
#     except Exception as e:
#         return f"Error accessing Chroma DB: {e}", [], []

#     if not results:
#         return "Chroma DB operation failed – no relevant information found.", [], []

#     docs_texts = [doc.page_content for doc in results]
#     doc_ids = [doc.metadata.get('id', 'N/A') for doc in results]

#     context = "\n\n".join(docs_texts)
#     user_prompt = f"original_user_query: {user_query}\ncontext:\n{context}"

#     final_prompt = f"{system_prompt}\n\nUser Query: {user_prompt}"

#     try:
#         ai_response = model.generate_content(final_prompt)
#         return ai_response.text, docs_texts, doc_ids
#     except Exception as e:
#         return f"Error generating Gemini response: {e}", docs_texts, doc_ids

# # ----------- Streamlit App -----------
# def main():
#     st.set_page_config(
#         page_title="Web Scraper & QA Bot",
#         page_icon="🤖",
#         layout="wide"
#     )
    
#     st.title("🤖 Web Scraper & QA Bot")
#     st.markdown("*Powered by Google Gemini & Chroma DB*")
    
#     # Sidebar for URL input and crawling
#     with st.sidebar:
#         st.header("🔗 Website Crawler")
#         url_input = st.text_input(
#             "Enter URL to crawl and index:",
#             placeholder="https://example.com"
#         )
        
#         if st.button("🚀 Start Crawling", type="primary"):
#             if url_input.strip():
#                 with st.spinner("Crawling website..."):
#                     progress_bar = st.progress(0)
#                     status_text = st.empty()
                    
#                     try:
#                         num_chunks = crawl_and_store(url_input, progress_bar, status_text)
#                         progress_bar.progress(1.0)
#                         status_text.text("✅ Crawling completed!")
#                         st.success(f"Successfully stored {num_chunks} chunks in Chroma DB!")
#                         st.session_state.crawling_done = True
#                     except Exception as e:
#                         st.error(f"Error during crawling: {e}")
#             else:
#                 st.warning("Please enter a valid URL")
    
#     # Main chat interface
#     st.header("💬 Ask Questions")
    
#     # Initialize chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
    
#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
            
#             # Show references for assistant messages
#             if message["role"] == "assistant" and "references" in message:
#                 with st.expander("📚 View References"):
#                     for i, ref in enumerate(message["references"]):
#                         st.text_area(
#                             f"Reference {i+1}:",
#                             value=ref,
#                             height=100,
#                             key=f"ref_{len(st.session_state.messages)}_{i}"
#                         )
    
#     # Chat input
#     if prompt := st.chat_input("Ask me anything about the crawled content..."):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate and display assistant response
#         with st.chat_message("assistant"):
#             with st.spinner("🧠 Searching in Chroma DB..."):
#                 response, references, doc_ids = query_bot(prompt)
            
#             st.markdown(response)
            
#             # Add assistant message to chat history
#             st.session_state.messages.append({
#                 "role": "assistant", 
#                 "content": response,
#                 "references": references
#             })
            
#             # Show references
#             if references:
#                 with st.expander("📚 View References"):
#                     for i, ref in enumerate(references):
#                         st.text_area(
#                             f"Reference {i+1}:",
#                             value=ref,
#                             height=100,
#                             key=f"ref_current_{i}"
#                         )
    
#     # Clear chat button
#     if st.button("🗑️ Clear Chat"):
#         st.session_state.messages = []
#         st.rerun()

# if __name__ == "__main__":
#     main()

import os
import uuid
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

import chromadb
import google.generativeai as genai
import streamlit as st
import json

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Google Generative AI model
@st.cache_resource
def get_model():
    return genai.GenerativeModel('gemini-1.5-flash')

# Setup Embedding Model
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

# Setup ChromaDB
@st.cache_resource
def get_chroma_client():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "website_crawling"
    embedding_model = get_embedding_model()
    
    return Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

def get_html_content(url, title='firstPage'):
    folder_path = f'./{title}'
    os.makedirs(folder_path, exist_ok=True)
    response = requests.get(url)
    with open(os.path.join(folder_path, 'output.html'), 'w', encoding='utf-8') as f:
        f.write(response.text)
    return response.text

def parse_html(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    page_title = soup.title.string if soup.title else 'No Title'
    page_content = soup.get_text(separator='\n', strip=True)
    
    links = []
    for a in soup.find_all('a', href=True):
        text = a.get_text(strip=True)
        href = a['href']
        full_url = urljoin(base_url, href)
        if full_url.startswith("http"):
            links.append({'text': text, 'url': full_url})
    
    return {
        'page_title': page_title,
        'page_content': page_content,
        'urls': links
    }

# ----------- 🧠 Crawling and Storing Embeddings -----------
def crawl_and_store(url, progress_bar=None, status_text=None):
    chroma_langchain = get_chroma_client()
    
    if status_text:
        status_text.text("Fetching main page...")
    
    html = get_html_content(url)
    website_data = parse_html(html, base_url=url)
    
    # Store main page content with metadata
    all_chunks = []
    all_metadatas = []
    
    # Process main page
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    main_chunks = splitter.split_text(website_data['page_content'])
    
    for chunk in main_chunks:
        all_chunks.append(chunk)
        all_metadatas.append({
            'source_url': url,
            'page_title': website_data['page_title'],
            'chunk_id': str(uuid.uuid4())
        })
    
    total_links = len(website_data['urls'])
    
    # Process linked pages
    for i, link in enumerate(website_data['urls']):
        try:
            if status_text:
                status_text.text(f"Crawling page {i+1}/{total_links}: {link['text'][:50]}...")
            
            sub_html = get_html_content(link['url'], title=f'page_{i}')
            sub_data = parse_html(sub_html, base_url=link['url'])
            
            # Split sub-page content
            sub_chunks = splitter.split_text(sub_data['page_content'])
            
            for chunk in sub_chunks:
                all_chunks.append(chunk)
                all_metadatas.append({
                    'source_url': link['url'],
                    'page_title': sub_data['page_title'],
                    'chunk_id': str(uuid.uuid4())
                })
            
            if progress_bar:
                progress_bar.progress((i + 1) / total_links)
                
        except Exception as e:
            if status_text:
                status_text.text(f"Skipping {link['url']}: {str(e)[:50]}...")
            continue

    if status_text:
        status_text.text("Processing and storing content...")

    # Store chunks in Chroma with metadata
    ids = [metadata['chunk_id'] for metadata in all_metadatas]
    chroma_langchain.add_texts(texts=all_chunks, metadatas=all_metadatas, ids=ids)
    
    return len(all_chunks)

# ----------- 🤖 QA Bot with Gemini + ChromaDB -----------
def optimize_query(user_query):
    """Enhance user query for better semantic search"""
    # Add context keywords to improve retrieval
    enhanced_query = user_query
    
    # Add common programming/technical terms if query seems technical
    technical_keywords = ['code', 'function', 'method', 'class', 'variable', 'API', 'documentation', 'tutorial', 'guide', 'example']
    if any(keyword in user_query.lower() for keyword in technical_keywords):
        enhanced_query += " programming code example implementation"
    
    # Add question context words for better matching
    question_words = ['how', 'what', 'why', 'when', 'where', 'which']
    if any(word in user_query.lower() for word in question_words):
        enhanced_query += " explanation guide tutorial"
    
    return enhanced_query

def get_diverse_results(user_query, chroma_langchain, k=5):
    """Get diverse, high-quality results using multiple search strategies"""
    
    # Strategy 1: Direct query
    direct_results = chroma_langchain.similarity_search(user_query, k=k)
    
    # Strategy 2: Enhanced query
    enhanced_query = optimize_query(user_query)
    enhanced_results = chroma_langchain.similarity_search(enhanced_query, k=k)
    
    # Strategy 3: Keyword extraction and search
    keywords = extract_keywords(user_query)
    keyword_results = []
    if keywords:
        keyword_query = " ".join(keywords)
        keyword_results = chroma_langchain.similarity_search(keyword_query, k=k)
    
    # Combine and deduplicate results
    all_results = direct_results + enhanced_results + keyword_results
    seen_ids = set()
    unique_results = []
    
    for result in all_results:
        result_id = result.metadata.get('chunk_id', str(hash(result.page_content)))
        if result_id not in seen_ids:
            seen_ids.add(result_id)
            unique_results.append(result)
    
    # Sort by relevance score if available, otherwise keep order
    return unique_results[:k]

def extract_keywords(text):
    """Extract key terms from user query"""
    import re
    # Remove common stop words and extract meaningful terms
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    # Extract words (alphanumeric + common programming symbols)
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    return keywords[:5]  # Return top 5 keywords

def query_bot(user_query):
    model = get_model()
    chroma_langchain = get_chroma_client()
    
    system_prompt = '''You are an expert technical assistant with deep knowledge across multiple domains. Your role is to provide comprehensive, accurate, and helpful answers based on the provided context from the knowledge base.

## Instructions:
1. **Answer Quality**: Provide detailed, well-structured responses that directly address the user's question
2. **Context Usage**: Use ONLY the information provided in the context - do not add external knowledge
3. **Code Formatting**: 
   - Format all code snippets using proper markdown code blocks with language specification
   - Use ```python, ```javascript, ```html, ```css, etc. as appropriate
   - For inline code, use `backticks`
4. **Structure**: Organize your response with clear headings, bullet points, and examples when helpful
5. **Accuracy**: If the context doesn't contain sufficient information, clearly state what information is missing
6. **No JSON**: Respond in natural language format, not JSON

## Response Guidelines:
- Start with a direct answer to the main question
- Provide step-by-step explanations when appropriate
- Include relevant code examples with proper formatting
- Add context and explanations for technical concepts
- End with additional helpful information if available

## If No Relevant Context:
Simply respond: "I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing your query or check if the content has been properly indexed."

Remember: Be conversational, helpful, and thorough while staying strictly within the bounds of the provided context.'''

    try:
        # Use optimized search strategy
        results = get_diverse_results(user_query, chroma_langchain, k=5)
    except Exception as e:
        return f"Error accessing Chroma DB: {e}", [], [], []

    if not results:
        return "I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing your query or check if the content has been properly indexed.", [], [], []

    # Filter and rank results by relevance
    filtered_results = []
    for result in results:
        # Basic relevance check - ensure the result contains some query terms
        query_terms = extract_keywords(user_query)
        content_lower = result.page_content.lower()
        
        # Calculate basic relevance score
        relevance_score = sum(1 for term in query_terms if term in content_lower)
        
        if relevance_score > 0 or len(filtered_results) < 2:  # Ensure we have at least 2 results
            filtered_results.append((result, relevance_score))
    
    # Sort by relevance score (descending) and take top 3
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    final_results = [result[0] for result in filtered_results[:3]]

    docs_texts = [doc.page_content for doc in final_results]
    source_urls = [doc.metadata.get('source_url', 'Unknown') for doc in final_results]
    page_titles = [doc.metadata.get('page_title', 'No Title') for doc in final_results]
    
    # Create rich context with clear separation
    context_parts = []
    for i, (url, title, text) in enumerate(zip(source_urls, page_titles, docs_texts), 1):
        context_parts.append(f"--- Source {i} ---\nURL: {url}\nPage Title: {title}\nContent:\n{text}\n")
    
    context = "\n".join(context_parts)
    
    user_prompt = f"""User Question: {user_query}

Context from Knowledge Base:
{context}

Please provide a comprehensive answer based on the above context."""

    final_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        ai_response = model.generate_content(final_prompt)
        return ai_response.text, docs_texts, source_urls, page_titles
    except Exception as e:
        return f"Error generating response: {e}", docs_texts, source_urls, page_titles

# ----------- Streamlit App -----------
def main():
    st.set_page_config(
        page_title="Web Scraper & QA Bot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Web Scraper & QA Bot")
    st.markdown("*Powered by Google Gemini & Chroma DB*")
    
    # Sidebar for URL input and crawling
    with st.sidebar:
        st.header("🔗 Website Crawler")
        url_input = st.text_input(
            "Enter URL to crawl and index:",
            placeholder="https://example.com"
        )
        
        if st.button("🚀 Start Crawling", type="primary"):
            if url_input.strip():
                with st.spinner("Crawling website..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        num_chunks = crawl_and_store(url_input, progress_bar, status_text)
                        progress_bar.progress(1.0)
                        status_text.text("✅ Crawling completed!")
                        st.success(f"Successfully stored {num_chunks} chunks in Chroma DB!")
                        st.session_state.crawling_done = True
                    except Exception as e:
                        st.error(f"Error during crawling: {e}")
            else:
                st.warning("Please enter a valid URL")
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show references for assistant messages
            if message["role"] == "assistant" and "references" in message:
                with st.expander("📚 View References & Sources"):
                    references = message["references"]
                    source_urls = message.get("source_urls", [])
                    page_titles = message.get("page_titles", [])
                    
                    for i, ref in enumerate(references):
                        st.markdown(f"**Reference {i+1}:**")
                        if i < len(source_urls) and i < len(page_titles):
                            st.markdown(f"🔗 **Source:** [{page_titles[i]}]({source_urls[i]})")
                        st.text_area(
                            f"Content:",
                            value=ref,
                            height=100,
                            key=f"ref_{len(st.session_state.messages)}_{i}"
                        )
                        if i < len(references) - 1:
                            st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the crawled content..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Searching in Chroma DB..."):
                response, references, source_urls, page_titles = query_bot(prompt)
            
            st.markdown(response)
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "references": references,
                "source_urls": source_urls,
                "page_titles": page_titles
            })
            
            # Show references with source URLs
            if references:
                with st.expander("📚 View References & Sources"):
                    for i, (ref, url, title) in enumerate(zip(references, source_urls, page_titles)):
                        st.markdown(f"**Reference {i+1}:**")
                        st.markdown(f"🔗 **Source:** [{title}]({url})")
                        st.text_area(
                            f"Content:",
                            value=ref,
                            height=100,
                            key=f"ref_current_{i}"
                        )
                        st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()