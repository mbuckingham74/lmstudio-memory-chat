
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
import requests
from bs4 import BeautifulSoup
import re

# Connect to your local LM Studio
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    model="qwen/qwen3-coder-30b"
)

def fetch_webpage(input_text: str) -> str:
    """Fetch and extract text content from a webpage. Input should be a URL."""
    # Extract URL from various formats
    url = input_text.strip()
    
    # Try to extract URL from JSON if that's what we got
    if '{' in url:
        import json
        try:
            parsed = json.loads(url)
            url = parsed.get('url', url)
        except:
            pass
    
    # Extract URL using regex as fallback
    url_match = re.search(r'https?://[^\s\'"}\]]+', url)
    if url_match:
        url = url_match.group(0)
    
    print(f"Attempting to fetch: {url}")
    
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        
        # Limit to first 3000 chars
        return text[:3000] if text else "No content found"
    except Exception as e:
        return f"Error fetching webpage: {str(e)}"

# Create tool
tools = [
    Tool(
        name="fetch_webpage",
        func=fetch_webpage,
        description="Useful for fetching and reading webpage content. Just pass the URL directly like: https://example.com"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    result = agent.invoke({"input": "Fetch https://example.com and tell me what it's about"})
    print(f"\nFinal Answer: {result['output']}")
