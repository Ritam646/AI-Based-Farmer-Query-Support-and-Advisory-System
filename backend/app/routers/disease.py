from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from ..main import handle_image_query, save_to_history, get_current_user

router = APIRouter()

@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    tgt_lang: str = Form("hin_Deva"),
    user_id: str = Depends(get_current_user)
):
    try:
        response = await handle_image_query(file, tgt_lang)
        await save_to_history(user_id, "image", file.filename, response["predicted_disease_en"])
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")