import gradio as gr
import os
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup
import re
import chromadb
import uuid
import base64
import json

# Initialize Chroma
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="engineering_memory")

# LM Studio connection
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234")

# GitHub authentication (optional - for private repos)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

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

# GitHub URL helpers
def is_github_url(url: str) -> bool:
    """Check if URL is a GitHub URL."""
    return 'github.com' in url or 'raw.githubusercontent.com' in url

def convert_github_url_to_raw(url: str) -> str:
    """Convert GitHub web URL to raw content URL."""
    # Already a raw URL
    if 'raw.githubusercontent.com' in url:
        return url

    # Convert github.com/user/repo/blob/branch/path to raw URL
    # Pattern: https://github.com/user/repo/blob/branch/path/to/file
    blob_match = re.match(
        r'https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)',
        url
    )
    if blob_match:
        user, repo, branch, path = blob_match.groups()
        return f'https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}'

    # Handle GitHub API URLs for repo contents
    # Pattern: https://github.com/user/repo (fetch README)
    repo_match = re.match(
        r'https?://github\.com/([^/]+)/([^/]+)/?$',
        url
    )
    if repo_match:
        user, repo = repo_match.groups()
        return f'https://api.github.com/repos/{user}/{repo}/readme'

    return url

def github_api_headers():
    """Get headers for GitHub API requests."""
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/vnd.github+json',
    }
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'
    return headers

def fetch_github_content(url: str) -> str:
    """Fetch content from GitHub with authentication."""
    headers = {'User-Agent': 'Mozilla/5.0'}

    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'

    raw_url = convert_github_url_to_raw(url)

    try:
        # Handle GitHub API endpoints differently
        if 'api.github.com' in raw_url:
            headers['Accept'] = 'application/vnd.github.raw'

        response = requests.get(raw_url, timeout=10, headers=headers)
        response.raise_for_status()

        # Return raw content (it's already text for code files)
        content = response.text
        return content[:5000] if content else "No content found"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"GitHub file not found: {url}"
        elif e.response.status_code == 401:
            return "GitHub authentication failed. Check your GITHUB_TOKEN."
        elif e.response.status_code == 403:
            return "GitHub access forbidden. Token may lack permissions or rate limit exceeded."
        return f"GitHub error: {str(e)}"
    except Exception as e:
        return f"Error fetching GitHub content: {str(e)}"

def parse_repo_url(repo_url: str) -> tuple:
    """Extract owner and repo from a GitHub URL."""
    # Match github.com/owner/repo patterns
    match = re.match(r'https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/.*)?$', repo_url)
    if match:
        return match.group(1), match.group(2)
    return None, None

def get_default_branch(owner: str, repo: str) -> str:
    """Get the default branch of a repository."""
    url = f'https://api.github.com/repos/{owner}/{repo}'
    response = requests.get(url, headers=github_api_headers(), timeout=10)
    response.raise_for_status()
    return response.json().get('default_branch', 'main')

def get_branch_sha(owner: str, repo: str, branch: str) -> str:
    """Get the SHA of a branch."""
    url = f'https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}'
    response = requests.get(url, headers=github_api_headers(), timeout=10)
    response.raise_for_status()
    return response.json()['object']['sha']

def create_branch(owner: str, repo: str, new_branch: str, from_branch: str = None) -> str:
    """Create a new branch from an existing branch."""
    if not GITHUB_TOKEN:
        return "Error: GITHUB_TOKEN required to create branches"

    try:
        # Get the source branch SHA
        if not from_branch:
            from_branch = get_default_branch(owner, repo)
        sha = get_branch_sha(owner, repo, from_branch)

        # Create new branch
        url = f'https://api.github.com/repos/{owner}/{repo}/git/refs'
        data = {
            'ref': f'refs/heads/{new_branch}',
            'sha': sha
        }
        response = requests.post(url, headers=github_api_headers(), json=data, timeout=10)
        response.raise_for_status()
        return f"Branch '{new_branch}' created from '{from_branch}'"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            return f"Branch '{new_branch}' already exists"
        return f"Error creating branch: {e.response.text}"
    except Exception as e:
        return f"Error creating branch: {str(e)}"

def create_or_update_file(owner: str, repo: str, path: str, content: str,
                          message: str, branch: str) -> str:
    """Create or update a file in a repository."""
    if not GITHUB_TOKEN:
        return "Error: GITHUB_TOKEN required to create/update files"

    try:
        url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'

        # Check if file exists to get its SHA
        file_sha = None
        try:
            response = requests.get(url, headers=github_api_headers(),
                                    params={'ref': branch}, timeout=10)
            if response.status_code == 200:
                file_sha = response.json().get('sha')
        except:
            pass

        # Create/update file
        data = {
            'message': message,
            'content': base64.b64encode(content.encode()).decode(),
            'branch': branch
        }
        if file_sha:
            data['sha'] = file_sha

        response = requests.put(url, headers=github_api_headers(), json=data, timeout=10)
        response.raise_for_status()

        action = "Updated" if file_sha else "Created"
        return f"{action} '{path}' on branch '{branch}'"
    except requests.exceptions.HTTPError as e:
        return f"Error updating file: {e.response.text}"
    except Exception as e:
        return f"Error updating file: {str(e)}"

def create_pull_request(owner: str, repo: str, title: str, body: str,
                        head_branch: str, base_branch: str = None) -> str:
    """Create a pull request."""
    if not GITHUB_TOKEN:
        return "Error: GITHUB_TOKEN required to create pull requests"

    try:
        if not base_branch:
            base_branch = get_default_branch(owner, repo)

        url = f'https://api.github.com/repos/{owner}/{repo}/pulls'
        data = {
            'title': title,
            'body': body,
            'head': head_branch,
            'base': base_branch
        }
        response = requests.post(url, headers=github_api_headers(), json=data, timeout=10)
        response.raise_for_status()

        pr_data = response.json()
        return f"PR #{pr_data['number']} created: {pr_data['html_url']}"
    except requests.exceptions.HTTPError as e:
        return f"Error creating PR: {e.response.text}"
    except Exception as e:
        return f"Error creating PR: {str(e)}"

# UI helper functions for GitHub operations
def ui_create_branch(repo_url: str, new_branch: str, from_branch: str) -> str:
    """UI wrapper for creating a branch."""
    if not repo_url.strip() or not new_branch.strip():
        return "Please provide repository URL and new branch name"

    owner, repo = parse_repo_url(repo_url)
    if not owner:
        return "Invalid GitHub repository URL"

    from_branch = from_branch.strip() if from_branch.strip() else None
    return create_branch(owner, repo, new_branch.strip(), from_branch)

def ui_commit_file(repo_url: str, branch: str, file_path: str,
                   file_content: str, commit_message: str) -> str:
    """UI wrapper for committing a file."""
    if not all([repo_url.strip(), branch.strip(), file_path.strip(), file_content.strip()]):
        return "Please fill in all required fields"

    owner, repo = parse_repo_url(repo_url)
    if not owner:
        return "Invalid GitHub repository URL"

    message = commit_message.strip() if commit_message.strip() else f"Update {file_path}"
    return create_or_update_file(owner, repo, file_path.strip(),
                                  file_content, message, branch.strip())

def ui_create_pr(repo_url: str, head_branch: str, base_branch: str,
                 pr_title: str, pr_body: str) -> str:
    """UI wrapper for creating a pull request."""
    if not all([repo_url.strip(), head_branch.strip(), pr_title.strip()]):
        return "Please provide repository URL, head branch, and PR title"

    owner, repo = parse_repo_url(repo_url)
    if not owner:
        return "Invalid GitHub repository URL"

    base = base_branch.strip() if base_branch.strip() else None
    return create_pull_request(owner, repo, pr_title.strip(),
                                pr_body.strip(), head_branch.strip(), base)

# Web fetching function
def fetch_webpage(url: str) -> str:
    url = url.strip()
    url_match = re.search(r'https?://[^\s\'"}\]]+', url)
    if url_match:
        url = url_match.group(0)

    # Use GitHub-specific fetcher for GitHub URLs
    if is_github_url(url):
        return fetch_github_content(url)

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

def respond(message, chat_history):
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
        bot_message = response.content
    except Exception as e:
        bot_message = f"Error: {str(e)}"
    
    chat_history.append([message, bot_message])
    return "", chat_history

# Build Gradio interface
with gr.Blocks(title="Local LLM Chat") as demo:
    gr.Markdown("# Local LLM Chat with Memory & Web Access")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(value=[], height=600)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me anything (include URLs to fetch webpages)...",
                    show_label=False,
                    scale=9
                )
                submit = gr.Button("Send", scale=1, variant="primary")

        with gr.Column(scale=1):
            with gr.Accordion("Memory Management", open=True):
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

            with gr.Accordion("GitHub Operations", open=False):
                gh_repo_url = gr.Textbox(
                    placeholder="https://github.com/owner/repo",
                    label="Repository URL"
                )
                gh_status = gr.Markdown()

                with gr.Tab("Create Branch"):
                    gh_new_branch = gr.Textbox(placeholder="feature/my-branch", label="New Branch Name")
                    gh_from_branch = gr.Textbox(placeholder="main (optional)", label="From Branch")
                    gh_create_branch_btn = gr.Button("Create Branch", variant="primary")

                with gr.Tab("Commit File"):
                    gh_branch = gr.Textbox(placeholder="feature/my-branch", label="Branch")
                    gh_file_path = gr.Textbox(placeholder="path/to/file.txt", label="File Path")
                    gh_file_content = gr.Textbox(placeholder="File contents...", label="Content", lines=5)
                    gh_commit_msg = gr.Textbox(placeholder="Update file", label="Commit Message")
                    gh_commit_btn = gr.Button("Commit File", variant="primary")

                with gr.Tab("Create PR"):
                    gh_pr_head = gr.Textbox(placeholder="feature/my-branch", label="Head Branch")
                    gh_pr_base = gr.Textbox(placeholder="main (optional)", label="Base Branch")
                    gh_pr_title = gr.Textbox(placeholder="Add new feature", label="PR Title")
                    gh_pr_body = gr.Textbox(placeholder="Description...", label="PR Description", lines=3)
                    gh_pr_btn = gr.Button("Create PR", variant="primary")

    # Event handlers
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])

    save_btn.click(add_memory, [memory_input], [memory_status])
    save_btn.click(lambda: "", None, [memory_input])

    search_btn.click(search_memory, [search_input], [search_results])

    # GitHub event handlers
    gh_create_branch_btn.click(
        ui_create_branch,
        [gh_repo_url, gh_new_branch, gh_from_branch],
        [gh_status]
    )
    gh_commit_btn.click(
        ui_commit_file,
        [gh_repo_url, gh_branch, gh_file_path, gh_file_content, gh_commit_msg],
        [gh_status]
    )
    gh_pr_btn.click(
        ui_create_pr,
        [gh_repo_url, gh_pr_head, gh_pr_base, gh_pr_title, gh_pr_body],
        [gh_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
