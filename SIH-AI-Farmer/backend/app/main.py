# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import os
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import SupabaseVectorStore
# from supabase import create_client, Client
# import pandas as pd
# from pathlib import Path
# import traceback
# import PyPDF2
# import docx
# from langchain_community.document_loaders import (
#     PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
# )

# # Initialize FastAPI app
# app = FastAPI()

# # Config (adjust paths and API keys)
# DATA_DIR = "./ai_farmer_database"  # Local path to your data folder
# GROQ_API_KEY = ""
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# # Initialize Supabase client
# # SUPABASE_URL = ""
# SUPABASE_URL = ""
# # SUPABASE_KEY = ""
# SUPABASE_KEY = ""
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# # Embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # ================== ROBUST LOADERS ==================
# class RobustPDFLoader(PyPDFLoader):
#     """Fallback to PyPDFLoader if fitz is unavailable."""
#     def load(self):
#         try:
#             return super().load()  # Use PyPDFLoader as the primary method
#         except Exception as e:
#             print(f"[PDF] PyPDFLoader failed for {self.file_path}: {e}\n{traceback.format_exc()}")
#             return []

# def load_excel_to_documents(file_path: str) -> list[Document]:
#     docs = []
#     try:
#         xls = pd.ExcelFile(file_path)
#         for sheet_name in xls.sheet_names:
#             df = xls.parse(sheet_name)
#             text = df.to_csv(index=False)
#             if text.strip():
#                 docs.append(Document(
#                     page_content=text,
#                     metadata={
#                         "source": str(file_path),
#                         "sheet": sheet_name,
#                         "loader": "pandas_excel"
#                     }
#                 ))
#     except Exception as e:
#         print(f"[EXCEL] Failed {file_path}: {e}\n{traceback.format_exc()}")
#     return docs

# def load_file_to_documents(file_path: str) -> list[Document]:
#     """Load a single file into Documents depending on type."""
#     ext = Path(file_path).suffix.lower()
#     try:
#         if ext == ".pdf":
#             return RobustPDFLoader(file_path).load()
#         elif ext in [".txt", ".md", ".log"]:
#             return TextLoader(file_path, autodetect_encoding=True).load()
#         elif ext == ".csv":
#             return CSVLoader(file_path).load()
#         elif ext in [".xls", ".xlsx"]:
#             return load_excel_to_documents(file_path)
#         elif ext == ".docx":
#             return Docx2txtLoader(file_path).load()
#         else:
#             print(f"[SKIP] Unsupported file type: {file_path}")
#             return []
#     except Exception as e:
#         print(f"[LOAD ERR] {file_path}: {e}\n{traceback.format_exc()}")
#         return []

# def gather_documents_recursive(root_dir: str, allowed_ext=None) -> list[Document]:
#     """Walk directory recursively, load all supported files into Documents."""
#     if allowed_ext is None:
#         allowed_ext = {".pdf", ".txt", ".csv", ".xls", ".xlsx", ".docx"}
#     all_docs = []
#     for dirpath, _, filenames in os.walk(root_dir):
#         for name in filenames:
#             ext = Path(name).suffix.lower()
#             if ext in allowed_ext:
#                 fpath = os.path.join(dirpath, name)
#                 docs = load_file_to_documents(fpath)
#                 for d in docs:
#                     d.metadata.setdefault("source", fpath)
#                     d.metadata["relpath"] = os.path.relpath(fpath, root_dir)
#                     d.metadata["subfolder"] = os.path.relpath(dirpath, root_dir)
#                 all_docs.extend(docs)
#     return all_docs

# # ================== RAG PIPELINE ==================
# def build_vectorstore(root_dir: str):
#     print(f"[INFO] Loading documents from {root_dir} ...")
#     docs = gather_documents_recursive(root_dir)

#     print(f"[INFO] Loaded {len(docs)} raw documents. Splitting ...")
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = splitter.split_documents(docs)

#     print(f"[INFO] Split into {len(chunks)} chunks. Building Supabase vector store ...")
#     vectorstore = SupabaseVectorStore.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         client=supabase,
#         table_name="documents"
#     )
#     return vectorstore

# def build_rag_chain(vectorstore, groq_api_key: str, model="llama-3.3-70b-versatile"):
#     llm = ChatGroq(api_key=groq_api_key, model=model)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",
#         return_source_documents=True
#     )
#     return qa_chain

# # ================== API ENDPOINTS ==================
# @app.on_event("startup")
# async def startup_event():
#     print("[STARTUP] Building vectorstore...")
#     vs = build_vectorstore(DATA_DIR)
#     app.state.rag_chain = build_rag_chain(vs, GROQ_API_KEY)

# class Query(BaseModel):
#     query: str

# @app.post("/rag")
# async def rag_query(q: Query):
#     try:
#         if not hasattr(app.state, "rag_chain"):
#             raise ValueError("RAG chain not initialized")
#         result = app.state.rag_chain.invoke({"query": q.query})
#         return {
#             "answer": result["result"],
#             "sources": [doc.metadata.get('relpath', doc.metadata.get('source')) for doc in result["source_documents"]]
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# def home():
#     return {"message": "RAG API Running"}




# last chance
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from supabase import create_client, Client
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI()

# Config
DATA_DIR = "./ai_farmer_database"
GROQ_API_KEY = ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
SUPABASE_URL = ""
SUPABASE_KEY = ""
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Disease prediction model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./all_crops_hybrid_best_model.pth"
class_names_path = "./classes.txt"

with open(class_names_path, 'r') as f:
    class_names = f.read().splitlines()

class ImprovedCNNViTHybrid(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ImprovedCNNViTHybrid, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.feature_dim = 2048
        self.patch_size = 7
        self.num_patches = 49
        self.embedding_dim = 768
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(self.feature_dim, self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embedding_dim))
        self.dropout = nn.Dropout(0.3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=8, dim_feedforward=2048, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        features = self.backbone(x)
        features = self.feature_projection(features)
        features = features.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)
        features = features + self.pos_embed
        features = self.dropout(features)
        encoded = self.transformer(features)
        cls_output = encoded[:, 0]
        return self.classifier(cls_output)

# Load disease model
disease_model = ImprovedCNNViTHybrid(num_classes=len(class_names))
state_dict = torch.load(MODEL_PATH, map_location=device)
disease_model.load_state_dict(state_dict['model_state_dict'])
disease_model.eval()
disease_model.to(device)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Document loading functions
class RobustPDFLoader(PyPDFLoader):
    def load(self):
        try:
            return super().load()
        except Exception as e:
            print(f"[PDF] PyPDFLoader failed for {self.file_path}: {e}")
            return []

def load_excel_to_documents(file_path: str) -> list[Document]:
    docs = []
    try:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            text = df.to_csv(index=False)
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": str(file_path), "sheet": sheet_name, "loader": "pandas_excel"}
                ))
    except Exception as e:
        print(f"[EXCEL] Failed {file_path}: {e}")
    return docs

def load_file_to_documents(file_path: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".pdf":
            return RobustPDFLoader(file_path).load()
        elif ext in [".txt", ".md", ".log"]:
            return TextLoader(file_path, autodetect_encoding=True).load()
        elif ext == ".csv":
            return CSVLoader(file_path).load()
        elif ext in [".xls", ".xlsx"]:
            return load_excel_to_documents(file_path)
        elif ext == ".docx":
            return Docx2txtLoader(file_path).load()
        else:
            print(f"[SKIP] Unsupported file type: {file_path}")
            return []
    except Exception as e:
        print(f"[LOAD ERR] {file_path}: {e}")
        return []

def gather_documents_recursive(root_dir: str, allowed_ext=None) -> list[Document]:
    if allowed_ext is None:
        allowed_ext = {".pdf", ".txt", ".csv", ".xls", ".xlsx", ".docx"}
    all_docs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in allowed_ext:
                fpath = os.path.join(dirpath, name)
                docs = load_file_to_documents(fpath)
                for d in docs:
                    d.metadata.setdefault("source", fpath)
                    d.metadata["relpath"] = os.path.relpath(fpath, root_dir)
                    d.metadata["subfolder"] = os.path.relpath(dirpath, root_dir)
                all_docs.extend(docs)
    return all_docs

# RAG pipeline
def build_vectorstore(root_dir: str):
    print(f"[INFO] Loading documents from {root_dir} ...")
    docs = gather_documents_recursive(root_dir)
    print(f"[INFO] Loaded {len(docs)} raw documents. Splitting ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks. Building Supabase vector store ...")
    vectorstore = SupabaseVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=supabase,
        table_name="documents"
    )
    return vectorstore

def build_rag_chain(vectorstore, groq_api_key: str, model="llama-3.3-70b-versatile"):
    llm = ChatGroq(api_key=groq_api_key, model=model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain

# API endpoints
@app.on_event("startup")
async def startup_event():
    print("[STARTUP] Building vectorstore...")
    vs = build_vectorstore(DATA_DIR)
    app.state.rag_chain = build_rag_chain(vs, GROQ_API_KEY)

class Query(BaseModel):
    query: str

@app.post("/rag")
async def rag_query(q: Query):
    try:
        if not hasattr(app.state, "rag_chain"):
            raise ValueError("RAG chain not initialized")
        result = app.state.rag_chain.invoke({"query": q.query})
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get('relpath', doc.metadata.get('source')) for doc in result["source_documents"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = val_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = disease_model(img)
            _, predicted = torch.max(output, 1)
            confidence = torch.softmax(output, dim=1)[0][predicted].item()
            predicted_class = class_names[predicted.item()]
        return {"disease": predicted_class, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
def home():
    return {"message": "RAG API Running"}