from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import jwt
from datetime import datetime
from .routers import disease, rag, market, transcription  # Import routers

load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client (use service role key for backend inserts)
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Include routers
app.include_router(disease.router, prefix="/disease")
app.include_router(rag.router, prefix="/rag")
app.include_router(market.router, prefix="/market")
app.include_router(transcription.router, prefix="/transcription")

# Unified query endpoint (handles text/image/voice)
@app.post("/query")
async def query(
    input_type: str = Form(...),
    tgt_lang: str = Form(...),
    text: str = Form(default=None),
    file: UploadFile = File(default=None),
    authorization: str = Header(None)  # Optional for anonymous
):
    user_id = "anonymous"
    if authorization:
        try:
            token = authorization.replace("Bearer ", "")
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub", "anonymous")
        except:
            pass
    try:
        if input_type == "text" and text:
            response = await handle_text_query(text, tgt_lang)
            await save_to_history(user_id, input_type, text, response["answer"])
            return response
        elif input_type == "image" and file:
            response = await handle_image_query(file, tgt_lang)
            await save_to_history(user_id, input_type, file.filename, response["predicted_disease_en"])
            return response
        elif input_type == "voice" and file:
            response = await handle_voice_query(file, tgt_lang)
            await save_to_history(user_id, input_type, file.filename, response["transcription"])
            return response
        else:
            raise HTTPException(status_code=400, detail="Invalid input type or missing input")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Helper functions (moved to main for simplicity; can be in utils)
async def save_to_history(user_id: str, input_type: str, input_data: str, response: str):
    try:
        supabase.table("user_history").insert({
            "user_id": user_id,
            "input_type": input_type,
            "input": input_data,
            "response": response,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        print(f"Error saving to history: {str(e)}")

async def handle_text_query(text: str, tgt_lang: str) -> dict:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text query cannot be empty")
    answer = f"Response for '{text}' (translated to {tgt_lang}): Red rust is a fungal disease caused by Cephaleuros virescens..."
    return {"answer": answer, "sources": ["Agricultural Knowledge Base"], "target_language": tgt_lang}

async def handle_image_query(file: UploadFile, tgt_lang: str) -> dict:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload an image.")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB")
    return {
        "predicted_disease_en": "Mango_Anthracnose",
        "predicted_disease_translated": "Mango_Anthracnose (translated)",
        "target_language": tgt_lang
    }

async def handle_voice_query(file: UploadFile, tgt_lang: str) -> dict:
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload an audio file.")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB")
    transcription = "Transcription: What is red rust in mango? Red rust is a fungal disease..."
    return {"transcription": transcription, "target_language": tgt_lang}

@app.get("/")
def home():
    return {"message": "AI Farmer Assistant Backend Running"}