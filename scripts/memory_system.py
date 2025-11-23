import chromadb
from chromadb.config import Settings
import requests
import uuid

# Initialize Chroma client (stores data locally)
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get a collection for your engineering notes
collection = client.get_or_create_collection(name="engineering_memory")

def add_memory(text, metadata=None):
    """Store a piece of information in memory"""
    memory_id = str(uuid.uuid4())
    collection.add(
        documents=[text],
        ids=[memory_id],
        metadatas=[metadata] if metadata else None
    )
    print(f"Memory stored: {text[:50]}...")

def search_memory(query, n_results=3):
    """Search for relevant memories based on a query"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    if results['documents'][0]:
        return results['documents'][0]
    else:
        return []

def ask_with_memory(question, lm_studio_url="http://localhost:1234/v1/chat/completions"):
    """Ask LM Studio a question with relevant memory as context"""
    # Search for relevant memories
    relevant_memories = search_memory(question, n_results=3)
    
    # Build context from memories
    context = "\n".join([f"- {mem}" for mem in relevant_memories]) if relevant_memories else "No relevant past context."
    
    # Create the prompt with memory context
    prompt = f"""Previous relevant context:
{context}

Current question: {question}"""
    
    # Send to LM Studio
    response = requests.post(lm_studio_url, json={
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    })
    
    answer = response.json()['choices'][0]['message']['content']
    return answer

# Test it out
if __name__ == "__main__":
    # Add some engineering memories
    print("Adding memories...")
    add_memory("Aluminum has a thermal expansion coefficient of 23.1 μm/m°C")
    add_memory("For the housing project, we're using 6061-T6 aluminum alloy")
    add_memory("The operating temperature range is -20°C to 85°C")
    
    # Ask a question that should retrieve relevant context
    print("\nAsking question with memory...")
    answer = ask_with_memory("What material are we using for the housing?")
    print(f"\nAnswer: {answer}")
