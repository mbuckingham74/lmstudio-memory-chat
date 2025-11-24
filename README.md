# Local LLM with Memory & Web Access

A containerized chat interface for local LLMs running in LM Studio, featuring persistent memory using ChromaDB and web browsing capabilities.

## Features

- 🤖 **Local LLM Integration** - Connects to LM Studio's API server
- 🧠 **Persistent Memory** - ChromaDB-based vector database for semantic memory
- 🌐 **Web Browsing** - Fetch and analyze webpage content
- 💬 **Clean UI** - Gradio-based chat interface
- 🐳 **Containerized** - Easy deployment with Docker Compose

## Prerequisites

- [LM Studio](https://lmstudio.ai/) installed and running
- [Docker](https://www.docker.com/) or [OrbStack](https://orbstack.dev/)
- A model loaded in LM Studio with the API server enabled

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Local_LLM.git
cd Local_LLM
```

2. **Start LM Studio's API server**
   - Open LM Studio
   - Load a model (e.g., qwen3-coder-30b)
   - Enable "Local LLM Service" in Settings > Developer
   - Start the server (default: http://localhost:1234)

3. **Launch the chat interface**
```bash
docker-compose up --build
```

4. **Open in browser**
   - Navigate to http://localhost:7860

## Usage

### Chat Interface
- Type questions in the chat input
- Include URLs in your messages to fetch and analyze webpages
- Memory is automatically searched for relevant context

### Memory Management (Sidebar)
- **Add Memory**: Save important facts, specifications, or notes
- **Search Memory**: Find previously stored information

### Example Queries
```
What's the tensile strength of tungsten?
Fetch https://example.com and summarize it
What material are we using for the housing project?
```

## Project Structure
```
Local_LLM/
├── docker-compose.yml    # Container orchestration
├── Dockerfile           # Container build instructions
├── requirements.txt     # Python dependencies
├── chat_ui.py          # Gradio chat interface
├── scripts/            # Standalone Python scripts
│   ├── memory_system.py
│   └── web_agent.py
└── chroma_db/          # Vector database (created on first run)
```

## Configuration

### Change LM Studio URL
Edit `docker-compose.yml`:
```yaml
environment:
  - LM_STUDIO_URL=http://host.docker.internal:YOUR_PORT
```

### Change Model
Edit `chat_ui.py`, line with model name:
```python
model="your-model-name"
```

## Standalone Scripts

The `scripts/` folder contains standalone Python scripts for non-containerized use:

### Memory System
```bash
python3 scripts/memory_system.py
```

### Web Agent
```bash
python3 scripts/web_agent.py
```

## Technical Details

- **Memory**: ChromaDB with `all-MiniLM-L6-v2` embeddings
- **Web Scraping**: BeautifulSoup4 + requests
- **LLM Integration**: LangChain with OpenAI-compatible API
- **UI Framework**: Gradio

## Troubleshooting

### Container can't reach LM Studio
- Ensure LM Studio server is running
- Check that `host.docker.internal` resolves (use `localhost` on Linux)

### Memory not persisting
- Check that `chroma_db/` folder is created and mounted properly
- Verify volume mount in `docker-compose.yml`

### Model not responding
- Confirm the model name matches your loaded model in LM Studio
- Check LM Studio server logs for errors

## Contributing

Feel free to open issues or submit pull requests!

## License

MIT

## Author

Built with assistance from Claude (Anthropic) and Adderall XR!
