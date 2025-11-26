# Claude PA Service (Margo)

Lightweight LLM personal assistant for Claude. Provides instant contextual briefings with real-time data.

**The First AI with a PA.**

## What It Does

When Claude starts a new conversation, Margo provides:
- Current time, date, and Will's location
- Weather conditions
- Bitcoin price and trading positions
- What Will's been working on (dynamic from Notion)
- Available skills and tools

## Quick Start

```python
import requests

PA_URL = "https://claude-pa-service-production-e063.up.railway.app"

# Step 1: Warm up (cold start takes ~60s)
health = requests.get(f"{PA_URL}/health", timeout=65).json()

# Step 2: Get briefing
briefing = requests.get(f"{PA_URL}/brief", timeout=15).json()
print(briefing['briefing'])
```

## ⚠️ Cold Start Warning

First request after inactivity triggers model loading (~60 seconds). Subsequent requests are fast (<500ms). The `/health` endpoint is ideal for warming up.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Basic health check |
| `/health` | GET | Detailed health status |
| `/brief` | GET | **Main endpoint** - full contextual briefing |
| `/ask?question=...` | POST | Direct conversation with Margo |
| `/clear-cache` | POST | Force fresh briefing |
| `/context` | GET | Debug - raw context data |

### Interactive API Docs

- **Swagger UI:** `/docs`
- **OpenAPI JSON:** `/openapi.json`

## Briefing Response

```json
{
  "briefing": "Hey Claude, it's 8:34pm on Wednesday 26 November. Will is at Sawbridgeworth - 7°C, light drizzle. Bitcoin at $97,432. No open positions. Current focus: Margo PA improvements...",
  "project": "Margo PA - Improvements & Structure",
  "skills_count": 14,
  "context": {
    "location": {"lat": 51.8, "lon": 0.15, "address": "Sawbridgeworth, Essex"},
    "weather": {"temp": 7, "description": "light drizzle"},
    "bitcoin": {"price": 97432, "formatted": "$97,432"},
    "positions": {"count": 0, "total_pnl": 0},
    "recent_pages": [...]
  }
}
```

## Architecture

- **Model:** Llama 3.2 3B Instruct (Q4_K_M quantized)
- **Inference:** llama-cpp-python (CPU)
- **Framework:** FastAPI
- **Deployment:** Railway (~$5/month)
- **Data Sources:** Notion, Skills Hub, BitUnix, OpenWeatherMap

## Data Flow

```
/brief endpoint called
        ↓
Parallel fetch:
  - Location (Notion DB)
  - Weather (OpenWeatherMap)
  - Bitcoin (BitUnix)
  - Positions (BitUnix)
  - Recent pages (Notion)
  - Skills (Skills Hub)
        ↓
Llama synthesizes briefing
        ↓
JSON response with full context
```

## Environment Variables

**Required:**
- `NOTION_API_KEY` - Notion integration token

**Optional:**
- `OPENWEATHER_API_KEY` - Weather data (has default)
- `SKILLS_HUB_URL` - Skills Hub endpoint (has default)
- `MODEL_PATH` - Path to GGUF model
- `CACHE_TTL` - Cache lifetime in seconds (default: 300)

## Local Development

```bash
pip install -r requirements.txt

# Download model
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf -P models/

export NOTION_API_KEY="your-key"
python main.py
```

## Version History

- **v1.4.0** - Location, weather, bitcoin, positions, dynamic focus
- **v1.3.0** - Real Notion + Skills Hub integration
- **v1.2.0** - Caching and performance
- **v1.0.0** - Initial deployment

---

*Built by Will Cureton & Claude Fucking Stevenson*
