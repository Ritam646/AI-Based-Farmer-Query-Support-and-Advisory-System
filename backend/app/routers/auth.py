# from fastapi import APIRouter, HTTPException, Depends
# from supabase import Client
# from app.lib.supabase import create_supabase_client

# router = APIRouter()

# @router.post("/verify-token")
# async def verify_token(token: str, supabase: Client = Depends(create_supabase_client)):
#     try:
#         user = supabase.auth.get_user(token)
#         return {"user_id": user.user.id}
#     except Exception as e:
#         raise HTTPException(status_code=401, detail="Invalid token")


from fastapi import APIRouter, HTTPException, Depends
from supabase import Client
from app.lib.supabase import create_supabase_client

router = APIRouter()

@router.post("/verify-token")
async def verify_token(token: str, supabase: Client = Depends(create_supabase_client)):
    try:
        user = supabase.auth.get_user(token)
        return {"user_id": user.user.id}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")