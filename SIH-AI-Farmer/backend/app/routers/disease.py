from fastapi import APIRouter, File, UploadFile, HTTPException
from app.lib.supabase import get_supabase_client
from app.models.disease_model import predict_disease
from PIL import Image
import time

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    supabase = get_supabase_client()
    try:
        img = Image.open(file.file).convert("RGB")
        result = predict_disease(img)
        user_id = supabase.auth.get_user().user.id if supabase.auth.get_user() else None
        if user_id:
            supabase.table("user_history").insert({
                "user_id": user_id,
                "input_type": "image",
                "input": file.filename,
                "response": str(result),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }).execute()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")