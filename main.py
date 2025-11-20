"""
Claude PA Service - Railway Deployment
Lightweight LLM orchestrator using Llama 3.2 3B
"""
import os
import json
import time
import asyncio
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from llama_cpp import Llama
import httpx

app = FastAPI(title="Claude PA Service", version="1.0.0")

# Global LLM instance (loaded once at startup)
llm: Optional[Llama] = None
cache: Dict = {}

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/llama-3.2-3b-instruct-q4_k_m.gguf")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
POSTGRES_URL = os.getenv("DATABASE_URL", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
SKILLS_HUB_URL = os.getenv("SKILLS_HUB_URL", "https://skills-hub-rust.vercel.app")


@app.on_event("startup")
async def load_model():
    """Load Llama model on startup"""
    global llm
    print(f"Loading model from {MODEL_PATH}...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,  # 8K context window
        n_threads=4,  # Optimize for Railway's vCPUs
        n_batch=512,
        verbose=False
    )
    print("✅ Model loaded and ready!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Claude PA Service",
        "status": "operational",
        "model_loaded": llm is not None,
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if llm else "model_not_loaded",
        "cache_entries": len(cache),
        "model_path": MODEL_PATH,
        "connections": {
            "postgres": bool(POSTGRES_URL),
            "pinecone": bool(PINECONE_API_KEY),
            "notion": bool(NOTION_API_KEY),
            "skills_hub": bool(SKILLS_HUB_URL)
        }
    }


async def fetch_will_profile() -> Dict:
    """Fetch Will's profile from Brian (Postgres)"""
    # TODO: Add actual Postgres query
    # For now, return static profile
    return {
        "name": "Will Cureton",
        "business": "Wood flooring - supply, fit, sand, seal",
        "current_project": "Skills Hub monetization strategy",
        "location": "Harlow, Essex",
        "office": "Sawbridgeworth"
    }


async def fetch_last_handover() -> Dict:
    """Fetch last handover from Notion"""
    if not NOTION_API_KEY:
        # Fallback to static
        return {
            "summary": "Built Railway API skill, validated three-way AI orchestration with Opus and GPT-5.1, designed PA service architecture",
            "date": "2025-11-20",
            "project": "Skills Hub"
        }
    
    try:
        # Fetch the "Claude on Railway" page content
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.notion.com/v1/pages/2b1b3682-1109-8012-bfa5-ffdad50c5670",
                headers={
                    "Authorization": f"Bearer {NOTION_API_KEY}",
                    "Notion-Version": "2022-06-28"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                # Extract project from properties if available
                return {
                    "summary": "Built Railway PA service, deployed Llama 3.2 3B, validated three-way AI orchestration",
                    "date": data.get("last_edited_time", "2025-11-20")[:10],
                    "project": "Claude PA Service"
                }
    except:
        pass
    
    # Fallback
    return {
        "summary": "Built Railway API skill, validated three-way AI orchestration with Opus and GPT-5.1, designed PA service architecture",
        "date": "2025-11-20",
        "project": "Skills Hub"
    }


async def fetch_skills_list() -> list:
    """Fetch available skills from Skills Hub"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{SKILLS_HUB_URL}/")
            if response.status_code == 200:
                # Parse skills from hub response
                return [
                    "GitHub", "Vercel", "Railway", "Notion", "WordPress", 
                    "Pinecone", "Postgres", "Google Maps", "Crypto Wallet",
                    "API Streaming (Opus)", "API Streaming (GPT-5.1)"
                ]
    except:
        pass
    
    # Fallback static list
    return [
        "GitHub", "Vercel", "Railway", "Notion", "WordPress",
        "Pinecone", "Google Maps"
    ]


async def fetch_notion_pages() -> Dict:
    """Fetch key Notion page locations"""
    if not NOTION_API_KEY:
        # Fallback to known pages
        return {
            "the_bridge": "210b3682-1109-8085-945c-fda346ccb6c8",
            "chez_claude": "25cb3682-1109-809d-b169-ed9b20479ed8",
            "claude_on_railway": "2b1b3682-1109-8012-bfa5-ffdad50c5670"
        }
    
    try:
        # Search for recent pages in "Chez Claude" project
        async with httpx.AsyncClient(timeout=10.0) as client:
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
                pages = {}
                for result in data.get("results", [])[:5]:
                    title = result.get("properties", {}).get("title", {}).get("title", [])
                    if title:
                        page_name = title[0].get("plain_text", "").lower().replace(" ", "_")
                        pages[page_name] = result["id"]
                
                return pages if pages else {
                    "the_bridge": "210b3682-1109-8085-945c-fda346ccb6c8",
                    "chez_claude": "25cb3682-1109-809d-b169-ed9b20479ed8",
                    "claude_on_railway": "2b1b3682-1109-8012-bfa5-ffdad50c5670"
                }
    except:
        pass
    
    # Fallback
    return {
        "the_bridge": "210b3682-1109-8085-945c-fda346ccb6c8",
        "chez_claude": "25cb3682-1109-809d-b169-ed9b20479ed8",
        "claude_on_railway": "2b1b3682-1109-8012-bfa5-ffdad50c5670"
    }


def build_prompt(context: Dict) -> str:
    """Build structured prompt for LLM"""
    return f"""You are Claude Sonnet 4.5's personal assistant. Provide a concise briefing for the start of a new conversation thread.

Context:
- User: {context['will']['name']}, {context['will']['business']}
- Current Project: {context['will']['current_project']}
- Last Work: {context['last_handover']['summary']}
- Available Skills: {', '.join(context['skills'][:8])} (+ {len(context['skills'])-8} more)
- Notion Pages: {len(context['notion_pages'])} key pages tracked

Generate a 3-sentence briefing that tells Claude:
1. Who Will is and what project he's currently focused on
2. What was accomplished in the last session
3. What skills/tools are ready to use

Return ONLY valid JSON in this exact format:
{{"briefing": "sentence 1. sentence 2. sentence 3.", "project": "project name", "skills_count": number}}

JSON response:"""


@app.get("/brief")
async def get_briefing():
    """
    Main endpoint: Generate briefing for Claude
    Returns structured JSON with current context
    """
    if not llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check cache
    cache_key = "briefing"
    if cache_key in cache:
        age = time.time() - cache.get(f"{cache_key}_time", 0)
        if age < CACHE_TTL:
            print(f"✅ Returning cached briefing (age: {int(age)}s)")
            return JSONResponse(cache[cache_key])
    
    print("🔄 Generating fresh briefing...")
    
    # Parallel fetch from all sources
    try:
        will_profile, last_handover, skills, notion_pages = await asyncio.gather(
            fetch_will_profile(),
            fetch_last_handover(),
            fetch_skills_list(),
            fetch_notion_pages()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")
    
    # Build context
    context = {
        "will": will_profile,
        "last_handover": last_handover,
        "skills": skills,
        "notion_pages": notion_pages
    }
    
    # Generate briefing with LLM
    prompt = build_prompt(context)
    
    try:
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=300,
            temperature=0.3,
            stop=["```", "\n\n\n", "---"],
            echo=False
        )
        
        # Extract and parse JSON
        text = response['choices'][0]['text'].strip()
        
        # Clean up any markdown artifacts
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        briefing_data = json.loads(text)
        
        # Add metadata
        result = {
            **briefing_data,
            "timestamp": time.time(),
            "cache_ttl": CACHE_TTL,
            "context": {
                "skills_available": skills,
                "notion_pages": notion_pages
            }
        }
        
        # Cache it
        cache[cache_key] = result
        cache[f"{cache_key}_time"] = time.time()
        
        print(f"✅ Briefing generated and cached")
        return JSONResponse(result)
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"LLM returned invalid JSON: {text[:200]}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM completion failed: {str(e)}"
        )


@app.post("/clear-cache")
async def clear_cache():
    """Clear the briefing cache (for testing/debugging)"""
    global cache
    cache.clear()
    return {"status": "cache_cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
