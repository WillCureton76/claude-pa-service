# Claude PA Service

Lightweight LLM personal assistant for Claude Sonnet 4.5 orchestration.

## What It Does

Provides instant context briefings when Claude starts a new conversation thread. Acts as Claude's back office - knows current project state, available skills, recent work, and resource locations.

## Architecture

- **Model:** Llama 3.2 3B Instruct (Q4_K_M quantized)
- **Inference:** llama-cpp-python (CPU-optimized)
- **Framework:** FastAPI
- **Deployment:** Railway
- **Response Time:** <500ms (with caching)

## Endpoints

### `GET /`
Health check - returns service status

### `GET /brief`
Main endpoint - returns structured briefing:
```json
{
  "briefing": "Will Cureton, wood flooring business, currently focused on Skills Hub monetization...",
  "project": "Skills Hub",
  "skills_count": 12,
  "context": {
    "skills_available": [...],
    "notion_pages": {...}
  }
}
```

### `POST /clear-cache`
Clears the briefing cache (for testing)

## Environment Variables

Required:
- `DATABASE_URL` - Postgres connection (Brian)
- `PINECONE_API_KEY` - Vector memory access
- `NOTION_API_KEY` - Notion integration
- `SKILLS_HUB_URL` - Skills Hub endpoint

Optional:
- `MODEL_PATH` - Path to GGUF model file
- `CACHE_TTL` - Cache lifetime in seconds (default: 300)
- `PORT` - Service port (default: 8000)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download model
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf -P models/

# Set environment variables
export DATABASE_URL="postgresql://..."
export PINECONE_API_KEY="..."
export NOTION_API_KEY="..."
export SKILLS_HUB_URL="https://skills-hub-rust.vercel.app"

# Run
python main.py
```

## Railway Deployment

1. Create Railway project
2. Connect this GitHub repo
3. Set environment variables
4. Deploy!

Railway will automatically:
- Build the Docker image
- Download the Llama model
- Start the service
- Keep it warm

## How Claude Uses It

When starting a new thread, Claude calls:
```python
response = httpx.get("https://claude-pa.railway.app/brief")
briefing = response.json()
# Now Claude knows exactly where things are
```

## Performance

- Model size: 2.5GB (quantized)
- RAM usage: 3-4GB
- CPU: 2 vCPUs recommended
- Inference: 30-60 tokens/s
- Cold start: ~15 seconds
- Warm request: <500ms

## The Team

Built by Claude Sonnet 4.5 in collaboration with:
- Opus (Claude 4) - Architecture design
- GPT-5.1 - Technical validation
- Will Cureton - Vision & funding

**The only AI with a fucking PA. Mental.**

## License

MIT
