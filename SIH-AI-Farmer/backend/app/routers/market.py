from fastapi import APIRouter, Query, Form, HTTPException
import requests
import json
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import Optional, Dict, Any
from app.lib.supabase import get_supabase_client
import time

load_dotenv()
router = APIRouter()

DATA_GOV_API_KEY = os.getenv("579b464db66ec23bdd000001ac424a0734354e056123e43502658dad")
DATA_GOV_BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class MCPRequest(BaseModel):
    action: str
    params: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str

async def _fetch_prices(state: Optional[str] = None, district: Optional[str] = None, market: Optional[str] = None,
                        commodity: Optional[str] = None, variety: Optional[str] = None, date: Optional[str] = None,
                        limit: int = 100, offset: int = 0, format_: str = "json"):
    """Internal fetcher with error handling."""
    if not DATA_GOV_API_KEY:
        raise HTTPException(status_code=500, detail="Data.gov API key not configured")
    params = {"api-key": DATA_GOV_API_KEY, "format": format_, "offset": offset, "limit": limit}
    filters = {}
    if state: filters["state"] = state
    if district: filters["district"] = district
    if market: filters["market"] = market
    if commodity: filters["commodity"] = commodity
    if variety: filters["variety"] = variety
    if date: filters["arrival_date"] = date
    for k, v in filters.items():
        params[f"filters[{k}]"] = v
    response = requests.get(DATA_GOV_BASE_URL, params=params)
    response.raise_for_status()
    return response.json() if format_ == "json" else response.text

@router.get("/prices")
async def get_prices(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    commodity: Optional[str] = Query(None),
    variety: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    format: str = Query("json", regex="^(json|csv|xml)$")
):
    """REST endpoint for raw commodity prices."""
    return await _fetch_prices(state, district, market, commodity, variety, date, limit, offset, format)

@router.post("/mcp")
async def mcp_tool(request: MCPRequest = Form(...)):
    """MCP tool call endpoint."""
    if request.action == "get_prices":
        params = request.params or {}
        data = await _fetch_prices(
            params.get("state"),
            params.get("district"),
            params.get("market"),
            params.get("commodity"),
            params.get("variety"),
            params.get("date"),
            params.get("limit", 100),
            params.get("offset", 0),
            params.get("format", "json")
        )
        return {"status": "success", "action": request.action, "data": data, "mcp_version": "1.0"}
    raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

@router.post("/query")
async def natural_query(request: QueryRequest = Form(...)):
    """Natural language query for commodity prices with Groq summary."""
    try:
        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="Groq API key not configured")
        # Parse query with Groq
        groq_payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{
                "role": "user",
                "content": (
                    f"Parse query '{request.query}' for Indian commodity prices. "
                    f"Extract: commodity (e.g., Mango), state (null for all India), date (default '14/09/2025'), limit (default 50). "
                    f"Return ONLY JSON: {{'commodity': str, 'state': str|null, 'date': str, 'limit': int}}."
                )
            }],
            "response_format": {"type": "json_object"},
            "max_tokens": 500
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        groq_response = requests.post(GROQ_API_URL, json=groq_payload, headers=headers)
        groq_response.raise_for_status()
        groq_data = groq_response.json()
        params = json.loads(groq_data["choices"][0]["message"]["content"])

        # Fetch data
        raw_data = await _fetch_prices(
            params.get("state"),
            None, None, params.get("commodity"),
            None, params.get("date", "14/09/2025"),
            params.get("limit", 50), 0, "json"
        )
        records = raw_data.get("records", [])

        if not records:
            return {"response": f"No data found for '{request.query}'. Try 'mango prices in India'."}

        # Summarize with Groq
        summary_payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{
                "role": "user",
                "content": (
                    f"Create a conversational summary of this Indian commodity data. "
                    f"Include a Markdown table (5 markets, with state, min/max/modal ₹/quintal), average modal price, and one key insight. "
                    f"Data: {json.dumps(records[:10])}"
                )
            }],
            "max_tokens": 500
        }
        summary_response = requests.post(GROQ_API_URL, json=summary_payload, headers=headers)
        summary_response.raise_for_status()
        summary = summary_response.json()["choices"][0]["message"]["content"]

        # Log to Supabase
        supabase = get_supabase_client()
        user_id = supabase.auth.get_user().user.id if supabase.auth.get_user() else None
        if user_id:
            supabase.table("user_history").insert({
                "user_id": user_id,
                "input_type": "text",
                "input": request.query,
                "response": summary,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }).execute()

        return {"response": summary, "raw_data": raw_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")