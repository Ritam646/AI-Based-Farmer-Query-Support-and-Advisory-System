# from fastapi import APIRouter, Depends, HTTPException
# from app.schemas import QueryRequest
# from app.utils.rag_setup import build_rag_chain
# from app.lib.supabase import create_supabase_client
# from supabase import Client
# import os

# router = APIRouter()

# # Initialize RAG chain at startup
# rag_chain = None

# @router.on_event("startup")
# async def startup_event():
#     global rag_chain
#     print("[STARTUP] Building RAG vectorstore...")
#     rag_chain = build_rag_chain("./data/ai_farmer_database", os.getenv("GROQ_API_KEY"))

# def rag_query(query: str):
#     if not rag_chain:
#         raise ValueError("RAG chain not initialized")
#     result = rag_chain.invoke({"query": query})
#     return {
#         "answer": result["result"],
#         "sources": [doc.metadata.get("relpath", doc.metadata.get("source")) for doc in result["source_documents"]]
#     }

# @router.post("/")
# async def query(q: QueryRequest):
#     try:
#         return rag_query(q.query)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException, Depends
from ..main import handle_text_query, save_to_history, get_current_user

router = APIRouter()

@router.post("/")
async def rag_query(
    query: str,
    user_id: str = Depends(get_current_user)
):
    try:
        response = await handle_text_query(query, "hin_Deva")
        await save_to_history(user_id, "text", query, response["answer"])
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")