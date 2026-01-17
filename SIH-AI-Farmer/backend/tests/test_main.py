import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Farmer Assistant Backend Running"}

def test_query_text():
    response = client.post("/query", data={"input_type": "text", "text": "Test query", "tgt_lang": "hin_Deva"})
    assert response.status_code == 200