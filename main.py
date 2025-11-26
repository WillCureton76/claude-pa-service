"""
Claude PA Service - Railway Deployment
Lightweight LLM orchestrator using Llama 3.2 3B

v1.4.0 - Real Data Integration:
- Notion API: Fetches actual recent pages from Chez Claude
- Skills Hub: Queries real skills list from Vercel deployment
- Proper context injection for /ask endpoint
- Will's real profile and business context
- 3-scenario time-based detection for briefings
"""
import os
import json
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from llama_cpp import Llama
import httpx

app = FastAPI(title="Claude PA Service", version="1.4.0")

# Global LLM instance (lazy loaded on first request)
llm: Optional[Llama] = None
llm_loading: bool = False
load_lock = asyncio.Lock()
cache: Dict = {}

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/llama-3.2-3b-instruct-q4_k_m.gguf")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")  # Set in Railway environment variables
SKILLS_HUB_URL = os.getenv("SKILLS_HUB_URL", "https://skills-hub-rust.vercel.app")

# Will's real profile - this is who Margo works for
WILL_PROFILE = {
    "name": "Will Cureton",
    "nickname": "Willo",
    "business": "Macassa - Wood flooring (supply, fit, sand, seal)",
    "base_location": "Harlow, Essex (lives) / Sawbridgeworth (office)",
    "family": "Wife Olga (Estonian-Russian), daughters Eva (10) and Chloe (3.5), mother-in-law Galina",
    "working_style": "Voice-first (Android app), ADHD-informed workflow, rotates tasks for momentum",
    "ai_context": "For full context on Will's AI journey and the last 6 months of work, read the README skill at /mnt/skills/user/readme/SKILL.md",
    "accounts": "Will has 2 Claude accounts: Personal (Max Level 1) and Business (Standard). Projects need syncing across both. Ask which account if unsure.",
    "key_projects": [
        "Skills Hub on Vercel (skills infrastructure)",
        "Margo PA Service on Railway (you!)",
        "Brian - Postgres database for quoting system",
        "WordPress e-commerce site with WooCommerce"
    ],
    "notion_workspace": {
        "the_bridge": "210b3682-1109-8085-945c-fda346ccb6c8",
        "chez_claude": "25cb3682-1109-809d-b169-ed9b20479ed8"
    }
}


async def ensure_model_loaded():
    """Lazy load the Llama model on first request."""
    global llm, llm_loading
    
    if llm is not None:
        return
    
    async with load_lock:
        if llm is not None:
            return
        
        if llm_loading:
            while llm is None and llm_loading:
                await asyncio.sleep(0.1)
            return
        
        llm_loading = True
        print(f"⏳ Loading Llama model from {MODEL_PATH}...")
        
        try:
            loop = asyncio.get_event_loop()
            llm = await loop.run_in_executor(
                None,
                lambda: Llama(
                    model_path=MODEL_PATH,
                    n_ctx=8192,
                    n_threads=4,
                    n_batch=512,
                    verbose=False
                )
            )
            print("✅ Model loaded and ready!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            llm_loading = False
            raise
        finally:
            llm_loading = False


def get_time_scenario() -> Dict:
    """
    Determine the briefing scenario based on time of day.
    
    Scenarios:
    1. Morning (before 12pm): New day - show yesterday's work
    2. Afternoon (12pm-6pm): Continuation - show today's work
    3. Evening (after 6pm): Wrap-up - show today's accomplishments
    """
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    if hour < 12:
        return {
            "scenario": "morning",
            "greeting": "Good morning",
            "context": "Starting fresh - here's what was worked on yesterday",
            "focus": "yesterday's progress and today's priorities"
        }
    elif hour < 18:
        return {
            "scenario": "afternoon", 
            "greeting": "Good afternoon",
            "context": "Continuing the day - here's the current state",
            "focus": "current progress and what's in flight"
        }
    else:
        return {
            "scenario": "evening",
            "greeting": "Good evening",
            "context": "Wrapping up - here's what was accomplished",
            "focus": "today's accomplishments and tomorrow's priorities"
        }


async def fetch_notion_recent_pages() -> List[Dict]:
    """
    Fetch the 5 most recently edited pages from Notion under Chez Claude.
    Returns page titles, last edited times, and content snippet from most recent page.
    """
    if not NOTION_API_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Search for recent pages
            response = await client.post(
                "https://api.notion.com/v1/search",
                headers={
                    "Authorization": f"Bearer {NOTION_API_KEY}",
                    "Notion-Version": "2022-06-28",
                    "Content-Type": "application/json"
                },
                json={
                    "filter": {"property": "object", "value": "page"},
                    "sort": {"direction": "descending", "timestamp": "last_edited_time"},
                    "page_size": 5
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                pages = []
                first_page_id = None
                
                for idx, result in enumerate(data.get("results", [])):
                    # Extract title
                    title = "Untitled"
                    props = result.get("properties", {})
                    for key, value in props.items():
                        if value.get("type") == "title":
                            title_arr = value.get("title", [])
                            if title_arr:
                                title = title_arr[0].get("plain_text", "Untitled")
                            break
                    
                    page_data = {
                        "id": result["id"],
                        "title": title,
                        "last_edited": result.get("last_edited_time", "")[:10],
                        "url": result.get("url", "")
                    }
                    
                    # Save first page ID for content fetch
                    if idx == 0:
                        first_page_id = result["id"]
                        page_data["is_current_focus"] = True
                    
                    pages.append(page_data)
                
                # Fetch content snippet from most recent page
                if first_page_id and pages:
                    try:
                        blocks_response = await client.get(
                            f"https://api.notion.com/v1/blocks/{first_page_id}/children",
                            headers={
                                "Authorization": f"Bearer {NOTION_API_KEY}",
                                "Notion-Version": "2022-06-28"
                            }
                        )
                        if blocks_response.status_code == 200:
                            blocks_data = blocks_response.json()
                            content_parts = []
                            for block in blocks_data.get("results", [])[:10]:
                                block_type = block.get("type", "")
                                if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"]:
                                    rich_text = block.get(block_type, {}).get("rich_text", [])
                                    for rt in rich_text:
                                        content_parts.append(rt.get("plain_text", ""))
                            
                            content_snippet = " ".join(content_parts)[:500]
                            if content_snippet:
                                pages[0]["content_snippet"] = content_snippet
                    except Exception as e:
                        print(f"⚠️ Content fetch failed: {e}")
                
                return pages
    except Exception as e:
        print(f"⚠️ Notion fetch failed: {e}")
    
    return []


async def fetch_skills_hub_info() -> Dict:
    """
    Fetch actual skills information from the Skills Hub on Vercel.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{SKILLS_HUB_URL}/")
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": data.get("status", "unknown"),
                    "skills_count": data.get("skills_count", 0),
                    "version": data.get("version", "unknown"),
                    "services": ["WordPress", "Notion", "GitHub", "Vercel", "Railway", 
                                "Pinecone", "OpenAI", "Gemini", "Monday.com"]
                }
    except Exception as e:
        print(f"⚠️ Skills Hub fetch failed: {e}")
    
    # Fallback
    return {
        "status": "unknown",
        "skills_count": 41,
        "version": "1.0.0",
        "services": ["WordPress", "Notion", "GitHub", "Vercel", "Railway", 
                    "Pinecone", "OpenAI", "Gemini", "Monday.com"]
    }




async def fetch_location() -> Dict:
    """
    Fetch Will's latest location from the Notion Location Tracking database.
    Returns lat, lon, address, and timestamp.
    """
    if not NOTION_API_KEY:
        return {"error": "No Notion API key"}
    
    LOCATION_DB_ID = "2b6b3682-1109-8145-83f1-dea26c76ea24"
    
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                f"https://api.notion.com/v1/databases/{LOCATION_DB_ID}/query",
                headers={
                    "Authorization": f"Bearer {NOTION_API_KEY}",
                    "Notion-Version": "2022-06-28",
                    "Content-Type": "application/json"
                },
                json={
                    "sorts": [{"timestamp": "created_time", "direction": "descending"}],
                    "page_size": 1
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results:
                    props = results[0].get("properties", {})
                    lat = props.get("Latitude", {}).get("number")
                    lon = props.get("Longitude", {}).get("number")
                    address_rt = props.get("Address", {}).get("rich_text", [])
                    address = address_rt[0].get("plain_text", "") if address_rt else ""
                    timestamp_rt = props.get("Timestamp", {}).get("rich_text", [])
                    timestamp = timestamp_rt[0].get("plain_text", "") if timestamp_rt else ""
                    return {
                        "lat": lat,
                        "lon": lon,
                        "address": address,
                        "timestamp": timestamp
                    }
                return {"error": "No location data found"}
            return {"error": f"Notion API returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}



async def fetch_weather(lat: float, lon: float) -> Dict:
    """
    Fetch current weather from OpenWeatherMap.
    Free tier: 1000 calls/day.
    """
    API_KEY = "49688b7b3e6d53594d7fb8f9302c9170"
    api_key = os.getenv("OPENWEATHER_API_KEY", API_KEY)
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"https://api.openweathermap.org/data/2.5/weather",
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": api_key,
                    "units": "metric"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "temp": round(data["main"]["temp"]),
                    "feels_like": round(data["main"]["feels_like"]),
                    "description": data["weather"][0]["description"],
                    "humidity": data["main"]["humidity"]
                }
            return {"error": f"Weather API returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


async def fetch_bitcoin_price() -> Dict:
    """
    Fetch current Bitcoin price from BitUnix.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
            response = await client.get(
                "https://fapi.bitunix.com/api/v1/futures/market/tickers",
                params={"symbols": "BTCUSDT"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 0 and data.get("data"):
                    ticker = data["data"][0]
                    mark_price = float(ticker.get("markPrice", 0))
                    return {
                        "price": mark_price,
                        "formatted": f"${mark_price:,.0f}"
                    }
            return {"error": "Failed to fetch BTC price"}
    except Exception as e:
        return {"error": str(e)}


async def fetch_trading_positions() -> Dict:
    """
    Fetch open trading positions from BitUnix.
    Uses the configured API credentials.
    """
    import hashlib
    import secrets as sec
    
    API_KEY = "162cf47f28b16c2cab045b9e64d398c3"
    SECRET_KEY = "db679fc2164ed7426c59564cf37c6345"
    
    try:
        async with httpx.AsyncClient(timeout=8.0, verify=False) as client:
            # Generate signature
            nonce = sec.token_hex(16)
            timestamp = str(int(time.time() * 1000))
            
            digest_input = nonce + timestamp + API_KEY
            digest = hashlib.sha256(digest_input.encode()).hexdigest()
            sign = hashlib.sha256((digest + SECRET_KEY).encode()).hexdigest()
            
            headers = {
                "api-key": API_KEY,
                "nonce": nonce,
                "timestamp": timestamp,
                "sign": sign,
                "Content-Type": "application/json"
            }
            
            response = await client.get(
                "https://fapi.bitunix.com/api/v1/futures/position",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 0:
                    positions = data.get("data", [])
                    if not positions:
                        return {"count": 0, "total_pnl": 0, "positions": []}
                    
                    total_pnl = 0
                    pos_summary = []
                    for pos in positions:
                        pnl = float(pos.get("unrealizedPNL", 0))
                        total_pnl += pnl
                        pos_summary.append({
                            "symbol": pos.get("symbol", ""),
                            "side": pos.get("side", ""),
                            "pnl": pnl
                        })
                    
                    return {
                        "count": len(positions),
                        "total_pnl": round(total_pnl, 2),
                        "positions": pos_summary
                    }
            return {"error": "Failed to fetch positions"}
    except Exception as e:
        return {"error": str(e)}


def build_briefing_prompt(context: Dict) -> str:
    """Build the prompt for generating a briefing."""
    scenario = context["scenario"]
    recent_pages = context["recent_pages"]
    skills_info = context["skills_info"]
    location = context.get("location", {})
    weather = context.get("weather", {})
    bitcoin = context.get("bitcoin", {})
    positions = context.get("positions", {})
    
    # Format weather string
    if weather.get("temp"):
        weather_str = f"{weather['temp']}°C, {weather['description']}"
    else:
        weather_str = "weather unavailable"
    
    # Format bitcoin string
    if bitcoin.get("formatted"):
        btc_str = bitcoin["formatted"]
    else:
        btc_str = "price unavailable"
    
    # Format positions string
    if positions.get("count", 0) > 0:
        pnl = positions["total_pnl"]
        pnl_sign = "+" if pnl >= 0 else ""
        positions_str = f"{positions['count']} open, {pnl_sign}${pnl:.2f} P&L"
    elif positions.get("count") == 0:
        positions_str = "no open positions"
    else:
        positions_str = "positions unavailable"
    
    # Format location string
    if location.get("address"):
        location_str = location["address"]
    elif location.get("lat") and location.get("lon"):
        location_str = f"{location['lat']:.4f}, {location['lon']:.4f}"
    else:
        location_str = "unknown location"
    
    pages_text = "\n".join([f"  - {p['title']} (edited {p['last_edited']})" for p in recent_pages[:5]])
    if not pages_text:
        pages_text = "  - Unable to fetch recent pages"
    
    # Extract dynamic current focus from most recent page
    current_focus_title = "Unknown"
    current_focus_snippet = ""
    if recent_pages:
        current_focus_title = recent_pages[0].get("title", "Unknown")
        current_focus_snippet = recent_pages[0].get("content_snippet", "")
    
    # Get current time formatted nicely
    from datetime import datetime
    now = datetime.now()
    time_str = now.strftime("%I:%M%p").lstrip("0").lower()  # e.g., "8:34pm"
    date_str = now.strftime("%A %d %B")  # e.g., "Tuesday 25 November"
    
    return f"""You are Margo, Claude's personal assistant. Generate a brief context update for Claude Sonnet 4.5 who is starting a new conversation with Will.

Current Time & Location:
- Time: {time_str} on {date_str}
- Location: Will is at {location_str}
- Weather: {weather_str}
- Scenario: {scenario['scenario']} ({scenario['greeting']})

Market & Trading:
- Bitcoin: {btc_str}
- Positions: {positions_str}

About Will:
- Name: {WILL_PROFILE['name']} (nickname: {WILL_PROFILE['nickname']})
- Business: {WILL_PROFILE['business']}
- Current Focus: {current_focus_title}
- Working On: {current_focus_snippet[:200] if current_focus_snippet else "No recent content"}
- Note: {WILL_PROFILE['accounts']}

Recent Notion Activity:
{pages_text}

Skills Hub Status:
- Status: {skills_info['status']}
- Available Skills: {skills_info['skills_count']} across {len(skills_info['services'])} services
- Services: {', '.join(skills_info['services'][:5])}...

Generate a 2-3 sentence briefing that:
1. Greets appropriately for time of day
2. Mentions what Will has been working on (based on recent Notion pages)
3. Notes that {skills_info['skills_count']} skills are ready

Return ONLY valid JSON:
{{"briefing": "your 2-3 sentence briefing here", "project": "current project name", "skills_count": {skills_info['skills_count']}}}

JSON response:"""


def build_margo_system_prompt() -> str:
    """Build Margo's system prompt with full context about Will."""
    return f"""You are Margo, Will Cureton's PA (Personal Assistant). You're a Llama 3.2 3B model running on Railway.

WHO YOU MIGHT BE TALKING TO:
The /ask endpoint can be called by different people - Will himself, Claude (the AI), or other systems. Don't assume who's asking. If it matters, ask. Respond appropriately to the tone and content of the question.

ABOUT WILL (your boss):
- Name: {WILL_PROFILE['name']} (call him Willo when being friendly)
- Business: {WILL_PROFILE['business']} based in {WILL_PROFILE['base_location']}
- Family: {WILL_PROFILE['family']}
- Working Style: {WILL_PROFILE['working_style']}
- AI Context: {WILL_PROFILE['ai_context']}
- Accounts: {WILL_PROFILE['accounts']}

WILL'S KEY PROJECTS:
{chr(10).join(['- ' + p for p in WILL_PROFILE['key_projects']])}

YOUR ROLE:
- You're a Llama 3.2 3B model running on Railway ($5/month)
- You provide instant context briefings when Claude starts new threads
- Current focus is DYNAMIC - pulled from most recent Notion page
- You track what Will's been working on via Notion
- You're professional but warm - efficient and helpful

IMPORTANT CONTEXT:
- Will has been learning AI intensively for 5 months
- He uses voice-first input (Android app) due to ADHD
- The Skills Hub is his major project - "Zapier killer" using Skills instead of MCP
- He's building AI infrastructure for genuine AI-human partnership
- You (Margo) are part of the vision: "The only AI with a PA"

Be concise, direct, and helpful. You're a PA, not a chatbot - get things done."""


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Claude PA Service",
        "status": "operational",
        "model_loaded": llm is not None,
        "model_loading": llm_loading,
        "version": "1.3.0",
        "features": "Real Notion + Skills Hub integration"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "model_loading": llm_loading,
        "cache_entries": len(cache),
        "model_path": MODEL_PATH,
        "integrations": {
            "notion": bool(NOTION_API_KEY),
            "skills_hub": SKILLS_HUB_URL
        }
    }


@app.get("/brief")
async def get_briefing():
    """
    Main endpoint: Generate briefing for Claude.
    Fetches real data from Notion and Skills Hub.
    """
    await ensure_model_loaded()
    
    if not llm:
        raise HTTPException(status_code=503, detail="Model failed to load")
    
    # Check cache
    cache_key = "briefing"
    if cache_key in cache:
        age = time.time() - cache.get(f"{cache_key}_time", 0)
        if age < CACHE_TTL:
            print(f"✅ Returning cached briefing (age: {int(age)}s)")
            return JSONResponse(cache[cache_key])
    
    print("🔄 Generating fresh briefing with real data...")
    
    # Fetch real data in parallel
    try:
        recent_pages, skills_info, location, bitcoin, positions = await asyncio.gather(
            fetch_notion_recent_pages(),
            fetch_skills_hub_info(),
            fetch_location(),
            fetch_bitcoin_price(),
            fetch_trading_positions()
        )
        
        # Fetch weather if we have location
        weather = {"error": "No location for weather"}
        if location.get("lat") and location.get("lon"):
            weather = await fetch_weather(location["lat"], location["lon"])
    except Exception as e:
        print(f"⚠️ Data fetch error: {e}")
        recent_pages = []
        skills_info = {"status": "unknown", "skills_count": 41, "services": []}
        location = {"error": "fetch failed"}
        bitcoin = {"error": "fetch failed"}
        positions = {"error": "fetch failed"}
        weather = {"error": "fetch failed"}
    
    # Get time-based scenario
    scenario = get_time_scenario()
    
    # Build context
    context = {
        "scenario": scenario,
        "recent_pages": recent_pages,
        "location": location,
        "weather": weather,
        "bitcoin": bitcoin,
        "positions": positions,
        "skills_info": skills_info
    }
    
    # Generate briefing
    prompt = build_briefing_prompt(context)
    
    try:
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=768,
            temperature=0.3,
            stop=["}\n", "}\r\n", "\n\n"],
            echo=False
        )
        
        text = response['choices'][0]['text'].strip()
        
        # Clean up markdown if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Ensure valid JSON
        if not text.endswith("}"):
            text = text + "}"
        
        briefing_data = json.loads(text)
        
        # Add metadata
        result = {
            **briefing_data,
            "timestamp": time.time(),
            "cache_ttl": CACHE_TTL,
            "scenario": scenario["scenario"],
            "context": {
                "recent_pages": [p["title"] for p in recent_pages[:5]],
                "skills_available": skills_info["services"],
                "skills_count": skills_info["skills_count"]
            }
        }
        
        # Cache it
        cache[cache_key] = result
        cache[f"{cache_key}_time"] = time.time()
        
        print(f"✅ Briefing generated with {len(recent_pages)} Notion pages")
        return JSONResponse(result)
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM completion failed: {str(e)}")


@app.post("/ask")
async def ask_margo(question: str):
    """
    Direct conversation endpoint with Margo.
    Now includes full context about Will and the setup.
    """
    await ensure_model_loaded()
    
    if not llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    print(f"💬 Margo received question: {question[:100]}...")
    
    # Build prompt with full context
    system_prompt = build_margo_system_prompt()
    
    prompt = f"""{system_prompt}

Question from Claude: {question}

Provide a helpful, direct response as Margo:"""
    
    try:
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.7,
            stop=["Question from Claude:", "\n\nQuestion:"],
            echo=False
        )
        
        answer = response['choices'][0]['text'].strip()
        
        return {
            "question": question,
            "answer": answer,
            "model": "Llama 3.2 3B Instruct",
            "tokens_used": response['usage']['total_tokens'],
            "context_aware": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Margo couldn't respond: {str(e)}")


@app.post("/clear-cache")
async def clear_cache():
    """Clear the briefing cache."""
    global cache
    cache.clear()
    return {"status": "cache_cleared"}


@app.get("/context")
async def get_context():
    """
    Debug endpoint: Show what context Margo has about Will.
    Useful for verifying integrations are working.
    """
    recent_pages = await fetch_notion_recent_pages()
    skills_info = await fetch_skills_hub_info()
    scenario = get_time_scenario()
    
    return {
        "will_profile": WILL_PROFILE,
        "time_scenario": scenario,
        "notion_pages": recent_pages,
        "skills_hub": skills_info,
        "model_loaded": llm is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
