import streamlit as st
import os
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup
import re
import chromadb
import uuid

# Initialize Chroma
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="engineering_memory")

# LM Studio connection
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234")

llm = ChatOpenAI(
    base_url=f"{LM_STUDIO_URL}/v1",
    api_key="not-needed",
    model="local-model"
)

# Memory functions
def add_memory(text, metadata=None):
    memory_id = str(uuid.uuid4())
    collection.add(
        documents=[text],
        ids=[memory_id],
        metadatas=[metadata] if metadata else None
    )
    return f"Memory stored: {text[:50]}..."

def search_memory(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    if results['documents'][0]:
        return results['documents'][0]
    return []

# Web fetching function
def fetch_webpage(url: str) -> str:
    url = url.strip()
    url_match = re.search(r'https?://[^\s\'"}\]]+', url)
    if url_match:
        url = url_match.group(0)
    
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        return text[:3000] if text else "No content found"
    except Exception as e:
        return f"Error fetching webpage: {str(e)}"

def ask_llm(prompt, memories=None):
    url_match = re.search(r'https?://[^\s]+', prompt)
    
    full_prompt = prompt
    
    if memories:
        context = "\n".join([f"- {mem}" for mem in memories])
        full_prompt = f"""Previous relevant context:
{context}

Current question: {prompt}"""
    
    if url_match:
        url = url_match.group(0)
        webpage_content = fetch_webpage(url)
        full_prompt = f"""{full_prompt}

Webpage content from {url}:
{webpage_content}"""
    
    response = llm.invoke(full_prompt)
    return response.content

# Custom CSS for centered layout
st.markdown("""
<style>
    /* Center the chat input when no messages */
    .stChatFloatingInputContainer {
        bottom: 50% !important;
        transform: translateY(50%);
    }
    
    /* Once messages exist, move it back to bottom */
    .has-messages .stChatFloatingInputContainer {
        bottom: 1rem !important;
        transform: none;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("🤖 Local LLM Chat with Memory & Web Access")

# Sidebar for memory management
with st.sidebar:
    st.header("Memory Management")
    
    with st.form("add_memory_form"):
        memory_text = st.text_area("Add to memory:")
        submit = st.form_submit_button("Save Memory")
        if submit and memory_text:
            result = add_memory(memory_text)
            st.success(result)
    
    st.divider()
    
    search_query = st.text_input("Search memories:")
    if search_query:
        memories = search_memory(search_query)
        if memories:
            st.write("**Found memories:**")
            for mem in memories:
                st.write(f"- {mem}")
        else:
            st.write("No memories found")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add class to body if messages exist
if len(st.session_state.messages) > 0:
    st.markdown('<div class="has-messages">', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything (include URLs to fetch webpages)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant_memories = search_memory(prompt, n_results=3)
            
            try:
                response = ask_llm(prompt, relevant_memories)
            except Exception as e:
                response = f"Error: {str(e)}"
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

if len(st.session_state.messages) > 0:
    st.markdown('</div>', unsafe_allow_html=True)
