import os
import requests
import uuid
import urllib.parse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from collections import defaultdict

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip().strip('"').strip("'")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().strip('"').strip("'").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip().strip('"').strip("'")

if not all([NVIDIA_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY]):
    raise Exception("Missing required environment variables: NVIDIA_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY")

EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
RERANK_MODEL = "nvidia/llama-nemotron-reranker-4b"
CHAT_MODEL = "meta/llama-3.1-70b-instruct"

API_BASE = "https://integrate.api.nvidia.com/v1"
EMBED_URL = f"{API_BASE}/embeddings"
RERANK_URL = "https://integrate.api.nvidia.com/v1/ranking"
CHAT_URL = f"{API_BASE}/chat/completions"

app = FastAPI()

conversation_store = defaultdict(list)
MAX_TURNS = 8

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

def get_query_embedding(text: str) -> list[float]:
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": [text],
        "model": EMBED_MODEL,
        "input_type": "query",
        "encoding_format": "float",
        "truncate": "NONE"
    }
    
    response = requests.post(EMBED_URL, headers=headers, json=payload, timeout=10)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"NVIDIA API Error: {response.text}")
        
    data = response.json()
    return data["data"][0]["embedding"]

def compute_rrf(vector_results, keyword_results, rrf_k=60):
    scores = {}
    # Process vector ranks
    for rank, item in enumerate(vector_results, start=1):
        scores[item['id']] = scores.get(item['id'], 0.0) + (1.0 / (rrf_k + rank))
    # Process keyword ranks
    for rank, item in enumerate(keyword_results, start=1):
        scores[item['id']] = scores.get(item['id'], 0.0) + (1.0 / (rrf_k + rank))
    
    # Merge items back based on highest RRF score
    all_items = {item['id']: item for item in vector_results + keyword_results}
    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [all_items[id] for id, _ in sorted_ids[:10]]

@app.post("/search")
def search_faq(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    try:
        query_vec = get_query_embedding(req.message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
        
    rpc_url = f"{SUPABASE_URL}/rest/v1/rpc/match_faq"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query_embedding": query_vec,
        "match_count": 5
    }
    
    try:
        response = requests.post(rpc_url, headers=headers, json=payload, timeout=10)
        if response.status_code != 200:
            raise Exception(f"RPC returned {response.status_code}: {response.text}")
        result_data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase RPC failed: {str(e)}")
        
    return {"candidates": result_data}

def rewrite_query(original_query: str, history: list) -> str:
    if not history:
        return original_query
        
    hist_text = ""
    for msg in history:
        hist_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
    prompt = f"""Given the following conversation history and the user's new message, rewrite the user's message into a standalone, context-rich English query that can be used for a database search. Do not answer the question, just rewrite it.
    
History:
{hist_text}

User Message: {original_query}
Standalone Query:"""

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.1
    }
    try:
        resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        pass
        
    return original_query

@app.post("/chat")
def chat_faq(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    session_id = req.session_id if req.session_id else str(uuid.uuid4())
    history = conversation_store[session_id]
    
    search_query = rewrite_query(req.message, history)
    print(f"--> Session: {session_id} | Rewritten Query: {search_query}")
        
    # Step 1: Embed User Query
    try:
        query_vec = get_query_embedding(search_query)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
        
    # Step 2A: Vector Search (Top 20)
    rpc_url = f"{SUPABASE_URL}/rest/v1/rpc/match_faq"
    rpc_headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json"
    }
    rpc_payload = {
        "query_embedding": query_vec,
        "match_count": 20
    }
    try:
        vector_resp = requests.post(rpc_url, headers=rpc_headers, json=rpc_payload, timeout=25)
        vector_candidates = vector_resp.json() if vector_resp.status_code == 200 else []
        if vector_resp.status_code != 200:
            print(f"--> Vector Error: {vector_resp.status_code} - {vector_resp.text}")
    except Exception as e:
        print(f"--> Vector Exception: {str(e)}")
        vector_candidates = []

    # Step 2B: Keyword Search (Top 20)
    import urllib.parse
    safe_query = urllib.parse.quote(search_query)
    keyword_url = f"{SUPABASE_URL}/rest/v1/faq_vectors?or=(question.ilike.*{safe_query}*,answer.ilike.*{safe_query}*)&limit=20"
    keyword_headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
    }
    try:
        keyword_resp = requests.get(keyword_url, headers=keyword_headers, timeout=25)
        keyword_candidates = keyword_resp.json() if keyword_resp.status_code == 200 else []
        if keyword_resp.status_code != 200:
            print(f"--> Keyword Error: {keyword_resp.status_code} - {keyword_resp.text}")
    except Exception as e:
        print(f"--> Keyword Exception: {str(e)}")
        keyword_candidates = []

    # Step 2C: Merge via Python RRF
    candidates = compute_rrf(vector_candidates, keyword_candidates)

    if not candidates:
        err_msg = "Supabase Connection Failed. "
        try:
            err_msg += f"URL Configured: {SUPABASE_URL[:15]}... "
            err_msg += f"Vector Status: {vector_resp.status_code} "
        except:
            pass
        return {
            "answer": f"Backend Error: {err_msg}. Please check Render Environment Variables.",
            "source": "fallback",
            "candidates": []
        }

    # Step 3: Rerank top 10
    passages = [f"Q: {c.get('question')} A: {c.get('answer')}" for c in candidates]
        
    rerank_headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    rerank_payload = {
        "model": RERANK_MODEL,
        "query": search_query,
        "documents": passages,
        "top_n": 3
    }
    
    try:
        # Try official endpoint
        rerank_resp = requests.post(RERANK_URL, headers=rerank_headers, json=rerank_payload, timeout=15)
        
        if rerank_resp.status_code == 200:
            rerank_data = rerank_resp.json()
            rankings = rerank_data.get("rankings", rerank_data.get("results", []))
            rankings.sort(key=lambda x: x.get("index", 0)) 
            top_3_indices = [r["index"] for r in rankings[:3]]
            best_candidates = [candidates[i] for i in top_3_indices]
        else:
            # If 404 or any error occurs, trigger Graceful Bypass
            print(f"--> Rerank API responded with {rerank_resp.status_code}. Bypassing reranker to keep system alive.")
            best_candidates = candidates[:3] # Direct pick top 3 from vector search

    except Exception as e:
        print(f"--> Rerank Exception: {str(e)}. Bypassing to vector search defaults.")
        best_candidates = candidates[:3] # Fallback to top 3 from vector search

    # Step 4: Build system prompt
    faq_context = ""
    for idx, bc in enumerate(best_candidates, 1):
        faq_context += f"{idx}. Q: {bc.get('question')} A: {bc.get('answer')}\n"
        
    sys_prompt = f"""Tu Nandaka Security App ka helpful assistant hai.
Neeche diye gaye FAQs ke basis pe user ke sawaal ka jawab de.
Sirf in FAQs ki information use kar. Agar jawab in FAQs me nahi hai, seedha bol de.
Jawab Hinglish me de (Hindi + English mix), friendly aur short rakho.
FAQs:
{faq_context}"""

    # Step 5: Call Chat LLM
    chat_headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    # Combine system prompt with history and new message
    messages = [{"role": "system", "content": sys_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": req.message})

    chat_payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    }
    
    try:
        chat_resp = requests.post(CHAT_URL, headers=chat_headers, json=chat_payload, timeout=45)
        if chat_resp.status_code != 200:
            raise Exception(f"Chat API failed: {chat_resp.text}")
            
        chat_data = chat_resp.json()
        answer = chat_data["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

    # Step 6: Update state store
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})
    
    # Truncate if needed (MAX_TURNS * 2 to account for both user and assistant)
    if len(history) > MAX_TURNS * 2:
        history = history[-(MAX_TURNS * 2):]
    conversation_store[session_id] = history

    return {
        "answer": answer,
        "session_id": session_id,
        "source": "llm",
        "candidates": [
            {
                "id": c.get("id"),
                "question": c.get("question"),
                "similarity": c.get("similarity", 0.0)
            } for c in best_candidates
        ]
    }
