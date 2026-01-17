# from fastapi import APIRouter, Query, Form, HTTPException
# import requests
# import json
# from typing import Optional, Dict, Any
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import os

# load_dotenv()
# router = APIRouter()

# DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")
# DATA_GOV_BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# class MCPRequest(BaseModel):
#     action: str
#     params: Optional[Dict[str, Any]] = None

# class QueryRequest(BaseModel):
#     query: str

# @router.get("/prices")
# async def get_prices(
#     state: Optional[str] = Query(None),
#     district: Optional[str] = Query(None),
#     market: Optional[str] = Query(None),
#     commodity: Optional[str] = Query(None),
#     variety: Optional[str] = Query(None),
#     date: Optional[str] = Query(None),
#     limit: int = Query(100, ge=1, le=10000),
#     offset: int = Query(0, ge=0),
#     format: str = Query("json", regex="^(json|csv|xml)$")
# ):
#     """REST endpoint for raw commodity prices."""
#     return await _fetch_prices(state, district, market, commodity, variety, date, limit, offset, format)

# @router.post("/mcp")
# async def mcp_tool(request: MCPRequest = Form(...)):
#     """MCP tool call endpoint."""
#     if request.action == "get_prices":
#         params = request.params or {}
#         data = await _fetch_prices(
#             params.get("state"),
#             params.get("district"),
#             params.get("market"),
#             params.get("commodity"),
#             params.get("variety"),
#             params.get("date"),
#             params.get("limit", 100),
#             params.get("offset", 0),
#             params.get("format", "json")
#         )
#         return {"status": "success", "action": request.action, "data": data, "mcp_version": "1.0"}
#     raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

# @router.post("/query")
# async def natural_query(request: QueryRequest = Form(...)):
#     """Natural language query for commodity prices with Groq summary."""
#     try:
#         # Parse query with Groq
#         groq_payload = {
#             "model": "llama-3.3-70b-versatile",
#             "messages": [{
#                 "role": "user",
#                 "content": (
#                     f"Parse query '{request.query}' for Indian commodity prices. "
#                     f"Extract: commodity (e.g., Mango), state (null for all India), date (default '14/09/2025'), limit (default 50). "
#                     f"Return ONLY JSON: {{'commodity': str, 'state': str|null, 'date': str, 'limit': int}}."
#                 )
#             }],
#             "response_format": {"type": "json_object"},
#             "max_tokens": 500
#         }
#         headers = {
#             "Authorization": f"Bearer {GROQ_API_KEY}",
#             "Content-Type": "application/json"
#         }
#         groq_response = requests.post(GROQ_API_URL, json=groq_payload, headers=headers)
#         groq_response.raise_for_status()
#         groq_data = groq_response.json()
#         params = json.loads(groq_data["choices"][0]["message"]["content"])

#         # Fetch data
#         raw_data = await _fetch_prices(
#             params.get("state"),
#             None, None, params.get("commodity"),
#             None, params.get("date", "14/09/2025"),
#             params.get("limit", 50), 0, "json"
#         )
#         records = raw_data.get("records", [])

#         if not records:
#             return {"response": f"No data found for '{request.query}'. Try 'mango prices in India'."}

#         # Summarize with Groq
#         summary_payload = {
#             "model": "llama-3.3-70b-versatile",
#             "messages": [{
#                 "role": "user",
#                 "content": (
#                     f"Create a conversational summary of this Indian commodity data. "
#                     f"Include a Markdown table (5 markets, with state, min/max/modal ₹/quintal), average modal price, and one key insight. "
#                     f"Data: {json.dumps(records[:10])}"
#                 )
#             }],
#             "max_tokens": 500
#         }
#         summary_response = requests.post(GROQ_API_URL, json=summary_payload, headers=headers)
#         summary_response.raise_for_status()
#         summary = summary_response.json()["choices"][0]["message"]["content"]

#         return {"response": summary, "raw_data": raw_data}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# async def _fetch_prices(state, district, market, commodity, variety, date, limit, offset, format_):
#     """Internal fetcher with error handling."""
#     params = {"api-key": DATA_GOV_API_KEY, "format": format_, "offset": offset, "limit": limit}
#     filters = {}
#     if state: filters["state"] = state
#     if district: filters["district"] = district
#     if market: filters["market"] = market
#     if commodity: filters["commodity"] = commodity
#     if variety: filters["variety"] = variety
#     if date: filters["arrival_date"] = date
#     for k, v in filters.items():
#         params[f"filters[{k}]"] = v
#     response = requests.get(DATA_GOV_BASE_URL, params=params)
#     response.raise_for_status()
#     return response.json() if format_ == "json" else response.text

from fastapi import APIRouter, HTTPException, Depends
from ..main import save_to_history, get_current_user

router = APIRouter()

@router.post("/query")
async def market_query(
    query: str,
    user_id: str = Depends(get_current_user)
):
    try:
        response = {
            "response": "| Market | State | Min Price | Max Price | Modal Price |\n|---|---|---|---|---|\n| Market A | State X | 100 | 150 | 120 |",
            "raw_data": {}
        }
        await save_to_history(user_id, "text", query, response["response"])
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")