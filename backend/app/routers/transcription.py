# from fastapi import APIRouter, UploadFile, File, Form
# from app.utils.translation import translate_text
# import tempfile
# import soundfile as sf
# from transformers import pipeline
# import torch
# import os

# router = APIRouter()

# # Whisper pipeline
# device = "cuda" if torch.cuda.is_available() else "cpu"
# whisper = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)

# @router.post("/transcribe")
# async def transcribe(file: UploadFile = File(...), tgt_lang: str = Form("hin_Deva")):
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(await file.read())
#             audio_path = tmp.name
#         audio_data, sample_rate = sf.read(audio_path)
#         result = whisper({"raw": audio_data, "sampling_rate": sample_rate})
#         transcribed = result["text"]
#         english_text = translate_text(transcribed, src_lang="hin_Deva", tgt_lang="eng_Latn")
#         os.remove(audio_path)
#         return {"transcribed_text": transcribed, "english_text": english_text}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from ..main import handle_voice_query, save_to_history, get_current_user

router = APIRouter()

@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    try:
        response = await handle_voice_query(file, "hin_Deva")
        await save_to_history(user_id, "voice", file.filename, response["transcription"])
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")