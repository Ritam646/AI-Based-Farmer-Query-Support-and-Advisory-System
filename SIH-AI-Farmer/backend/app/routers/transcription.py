from fastapi import APIRouter, File, UploadFile, HTTPException
from app.lib.supabase import get_supabase_client
from app.models.disease_model import transcribe_audio  # Placeholder
import time

router = APIRouter()

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    supabase = get_supabase_client()
    try:
        result = transcribe_audio(file.file)
        user_id = supabase.auth.get_user().user.id if supabase.auth.get_user() else None
        if user_id:
            supabase.table("user_history").insert({
                "user_id": user_id,
                "input_type": "voice",
                "input": file.filename,
                "response": str(result),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }).execute()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {str(e)}")