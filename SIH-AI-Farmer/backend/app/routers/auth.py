from fastapi import APIRouter, HTTPException
from app.lib.supabase import get_supabase_client

router = APIRouter()

@router.post("/login")
async def login(email: str, password: str):
    supabase = get_supabase_client()
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return {"user": response.user.id, "message": "Login successful"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Login failed: {str(e)}")