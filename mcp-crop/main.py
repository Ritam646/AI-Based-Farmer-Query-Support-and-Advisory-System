from fastapi import FastAPI, Query, Body, HTTPException
import requests
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel

app = FastAPI(
    title="Commodity Prices MCP Server with Groq",
    description="Server for natural-language commodity price queries across India, powered by Groq's free API.",
    version="1.0.7"
)

# data.gov.in API key
DATA_GOV_API_KEY = ""
DATA_GOV_BASE_URL = ""
# Groq API key (your provided key)
GROQ_API_KEY = ""
GROQ_API_URL = ""  # Fixed: Added /openai

class MCPRequest(BaseModel):
    action: str
    params: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str

@app.get("/prices")
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
    """REST endpoint for raw prices."""
    return await _fetch_prices(state, district, market, commodity, variety, date, limit, offset, format)

@app.post("/mcp")
async def mcp_tool(request: MCPRequest = Body(...)):
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

@app.post("/query")
async def natural_query(request: QueryRequest = Body(...)):
    """Main endpoint: User gives query, gets Groq answer."""
    try:
        # Parse with Groq (HTTP request)
        groq_payload = {
            "model": "llama-3.3-70b-versatile",  # Supported free model
            "messages": [{
                "role": "user",
                "content": (
                    f"Parse query '{request.query}' for Indian commodity prices. "
                    f"Extract: commodity (e.g., Rice), state (null for all India), date (default '14/09/2025'), limit (default 50). "
                    f"Return ONLY JSON: {{'commodity': str, 'state': str|null, 'date': str, 'limit': int}}."
                )
            }],
            "response_format": {"type": "json_object"},
            "max_tokens": 500  # Added: Safe limit (use max_tokens for compatibility)
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        groq_response = requests.post(GROQ_API_URL, json=groq_payload, headers=headers)
        if groq_response.status_code != 200:
            raise ValueError(f"Groq parse error {groq_response.status_code}: {groq_response.text}")
        groq_data = groq_response.json()
        params_str = groq_data["choices"][0]["message"]["content"]
        params = json.loads(params_str)

        # Fetch data
        raw_data = await _fetch_prices(
            params.get("state"),
            None, None, params.get("commodity"),
            None, params.get("date", "14/09/2025"),
            params.get("limit", 50), 0, "json"
        )
        records = raw_data.get("records", [])

        if not records:
            return {"response": f"No data for '{request.query}'. Try 'wheat prices in India'."}

        # Summarize with Groq
        summary_payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{
                "role": "user",
                "content": (
                    f"Create a conversational summary of this Indian commodity data. "
                    f"Include Markdown table (5 markets, with state, min/max/modal ₹/quintal), average modal price, and insight. "
                    f"Data: {json.dumps(records[:10])}"
                )
            }],
            "max_tokens": 500  # Added: Safe limit
        }
        summary_response = requests.post(GROQ_API_URL, json=summary_payload, headers=headers)
        if summary_response.status_code != 200:
            raise ValueError(f"Groq summary error {summary_response.status_code}: {summary_response.text}")
        summary_data = summary_response.json()
        summary = summary_data["choices"][0]["message"]["content"]

        return {"response": summary, "raw_data": raw_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

async def _fetch_prices(state, district, market, commodity, variety, date, limit, offset, format_):
    """Internal fetcher with error handling."""
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
    if response.status_code != 200:
        raise ValueError(f"Data.gov.in error {response.status_code}: {response.text}")
    return response.json() if format_ == "json" else response.text

@app.get("/")
async def root():
    return {"message": "Server ready! POST to /query with {'query': 'your question'} for answers."}