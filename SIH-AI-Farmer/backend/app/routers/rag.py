from fastapi import APIRouter, Form, HTTPException
from app.lib.supabase import get_supabase_client
from app.utils.rag_setup import rag_query
import time

router = APIRouter()

@router.post("/query")
async def query(text: str = Form(...)):
    supabase = get_supabase_client()
    try:
        result = rag_query(text)
        user_id = supabase.auth.get_user().user.id if supabase.auth.get_user() else None
        if user_id:
            supabase.table("user_history").insert({
                "user_id": user_id,
                "input_type": "text",
                "input": text,
                "response": str(result),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }).execute()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"RAG query failed: {str(e)}")

