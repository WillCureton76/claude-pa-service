"""
Claude PA Service - Clean Implementation
Lightweight LLM PA service for Claude orchestration
"""

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Global variables
llm = None
executor = None

def load_model():
    """Load the LLM model (runs in thread pool)"""
    try:
        from llama_cpp import Llama
        
        model_path = "/app/models/llama-3.2-3b-instruct-q4_k_m.gguf"
        
        print("🤖 Loading Llama 3.2 3B model...")
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            verbose=False
        )
        print("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def run_llm_sync(prompt: str, max_tokens: int = 300) -> Dict[str, Any]:
    """Run LLM inference synchronously (for thread pool)"""
    if not llm:
        raise Exception("Model not loaded")
    
    try:
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["Human:", "\n\nHuman:", "User:", "\n\nUser:"],
            echo=False
        )
        
        return {
            "text": response['choices'][0]['text'].strip(),
            "tokens": response['usage']['total_tokens']
        }
        
    except Exception as e:
        raise Exception(f"LLM inference failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown"""
    global llm, executor
    
    # Startup
    print("🚀 Starting Claude PA Service...")
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")
    
    # Load model in background
    loop = asyncio.get_event_loop()
    llm = await loop.run_in_executor(executor, load_model)
    
    if llm:
        print("✅ Service ready!")
    else:
        print("❌ Service started but model failed to load")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    if executor:
        executor.shutdown(wait=True)

# Create FastAPI app
app = FastAPI(
    title="Claude PA Service",
    description="Lightweight LLM PA for Claude orchestration",
    version="2.0.0",
    lifespan=lifespan
)

# Basic endpoints
@app.get("/")
async def root():
    """Service information"""
    return {
        "service": "Claude PA Service",
        "version": "2.0.0",
        "status": "operational",
        "model_loaded": llm is not None
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy" if llm else "loading",
        "model_loaded": llm is not None,
        "version": "2.0.0"
    }

# LLM endpoints
@app.get("/brief")
async def brief():
    """Get a briefing from Margo"""
    if not llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    prompt = """You are Margo, Claude Sonnet 4.5's personal assistant. You help Claude stay organized and informed about Will Cureton's projects.

Provide a brief status update for Claude. Be concise and professional:"""
    
    # Run LLM in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, run_llm_sync, prompt, 200)
        
        return {
            "briefing": result["text"],
            "project": "Will Cureton - AI Projects",
            "skills_count": 22,
            "tokens_used": result["tokens"],
            "model": "Llama 3.2 3B Instruct"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Briefing failed: {e}")

@app.post("/ask")
async def ask_margo(question: str):
    """Chat with Margo"""
    if not llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    prompt = f"""You are Margo, Claude Sonnet 4.5's personal assistant. You help keep Claude organized and informed about Will Cureton's projects, priorities, and available tools.

You are professional, concise, and helpful. You have a warm but efficient personality.

Question: {question}

Response:"""
    
    # Run LLM in thread pool
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, run_llm_sync, prompt, 150)
        
        return {
            "question": question,
            "answer": result["text"],
            "model": "Llama 3.2 3B Instruct",
            "tokens_used": result["tokens"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

# Development server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
