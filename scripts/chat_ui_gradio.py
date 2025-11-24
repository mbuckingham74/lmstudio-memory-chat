import gradio as gr
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
def add_memory(text):
    if not text.strip():
        return "Please enter some text to save."
    memory_id = str(uuid.uuid4())
    collection.add(
        documents=[text],
        ids=[memory_id]
    )
    return f"✓ Memory saved: {text[:50]}..."

def search_memory(query, n_results=3):
    if not query.strip():
        return "Please enter a search query."
    results = collection.query(query_texts=[query], n_results=n_results)
    if results['documents'][0]:
        found = "\n".join([f"• {mem}" for mem in results['documents'][0]])
        return f"**Found memories:**\n{found}"
    return "No memories found."

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

def chat(message, history):
    # Check for relevant memories
    relevant_memories = collection.query(query_texts=[message], n_results=3)
    memories = relevant_memories['documents'][0] if relevant_memories['documents'][0] else []
    
    # Build prompt
    full_prompt = message
    
    if memories:
        context = "\n".join([f"- {mem}" for mem in memories])
        full_prompt = f"""Previous relevant context:
{context}

Current question: {message}"""
    
    # Check for URL
    url_match = re.search(r'https?://[^\s]+', message)
    if url_match:
        url = url_match.group(0)
        webpage_content = fetch_webpage(url)
        full_prompt = f"""{full_prompt}

Webpage content from {url}:
{webpage_content}"""
    
    try:
        response = llm.invoke(full_prompt)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# Build Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Local LLM Chat") as demo:
    gr.Markdown("# 🤖 Local LLM Chat with Memory & Web Access")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(
                placeholder="Ask me anything (include URLs to fetch webpages)...",
                show_label=False,
                container=False
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### Memory Management")
            
            with gr.Group():
                gr.Markdown("**Add to memory:**")
                memory_input = gr.Textbox(
                    placeholder="Enter information to remember...",
                    show_label=False,
                    lines=3
                )
                save_btn = gr.Button("Save Memory", variant="primary")
                memory_status = gr.Markdown()
            
            gr.Markdown("---")
            
            with gr.Group():
                gr.Markdown("**Search memories:**")
                search_input = gr.Textbox(
                    placeholder="Search...",
                    show_label=False
                )
                search_btn = gr.Button("Search")
                search_results = gr.Markdown()
    
    # Event handlers
    msg.submit(chat, [msg, chatbot], [chatbot])
    msg.submit(lambda: "", None, [msg])
    
    save_btn.click(add_memory, [memory_input], [memory_status])
    save_btn.click(lambda: "", None, [memory_input])
    
    search_btn.click(search_memory, [search_input], [search_results])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
