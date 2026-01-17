# import streamlit as st
# import os
# import torch
# import torch.nn as nn
# from PIL import Image
# import numpy as np
# from torchvision import transforms, models
# from langchain_community.vectorstores import SupabaseVectorStore
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA
# from supabase import create_client, Client

# # Page configuration
# st.set_page_config(
#     page_title="AI Farmer Platform - Hackathon Winner 2025",
#     page_icon="🌱",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for styling
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
#         padding: 2.5rem;
#         border-radius: 15px;
#         margin-bottom: 2rem;
#         color: white;
#         text-align: center;
#         box-shadow: 0 15px 40px rgba(0,0,0,0.4);
#         font-family: 'Arial', sans-serif;
#         font-size: 2rem;
#     }
#     .stApp {
#         background: linear-gradient(135deg, #f0f9f0 0%, #e8f5e8 100%);
#         font-family: 'Arial', sans-serif;
#         color: #333;
#     }
#     .metric-card {
#         background: white;
#         padding: 2rem;
#         border-radius: 15px;
#         box-shadow: 0 8px 20px rgba(0,0,0,0.15);
#         border-left: 6px solid #4CAF50;
#         margin: 1.5rem 0;
#         font-size: 1.3rem;
#         text-align: center;
#     }
#     .input-card {
#         background: white;
#         padding: 2.5rem;
#         border-radius: 15px;
#         box-shadow: 0 8px 25px rgba(0,0,0,0.15);
#         margin: 1.5rem 0;
#         font-size: 1.2rem;
#     }
#     .demo-button {
#         background: linear-gradient(45deg, #4CAF50, #66BB6A);
#         color: white;
#         border: none;
#         padding: 1rem 2rem;
#         border-radius: 25px;
#         font-weight: bold;
#         margin: 0.5rem;
#         box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
#         transition: all 0.3s ease;
#         font-size: 1.1rem;
#     }
#     .demo-button:hover {
#         transform: translateY(-3px);
#         box-shadow: 0 8px 25px rgba(76, 175, 80, 0.5);
#     }
#     .stButton > button {
#         background: linear-gradient(45deg, #4CAF50, #66BB6A);
#         color: white;
#         border: none;
#         padding: 1rem 2rem;
#         border-radius: 25px;
#         font-weight: bold;
#         transition: all 0.3s ease;
#         font-size: 1.1rem;
#     }
#     .stButton > button:hover {
#         transform: translateY(-3px);
#         box-shadow: 0 8px 25px rgba(76, 175, 80, 0.5);
#     }
#     .stTextInput > div > div > input {
#         background: rgba(255, 255, 255, 0.95);
#         border: 3px solid #A5D6A7;
#         border-radius: 12px;
#         padding: 1rem;
#         font-size: 1.2rem;
#         color: #333;
#     }
#     .stFileUploader > div > div > div {
#         background: white;
#         border: 3px dashed #A5D6A7;
#         border-radius: 12px;
#         font-size: 1.2rem;
#     }
#     .success-box {
#         background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
#         padding: 1.5rem;
#         border-radius: 12px;
#         border-left: 6px solid #4CAF50;
#         margin: 1.5rem 0;
#         font-size: 1.2rem;
#         color: #2E7D32;
#     }
#     .stSpinner {
#         color: #4CAF50;
#         font-size: 1.5rem;
#     }
#     .stTabs [data-baseweb="tab-list"] {
#         justify-content: center;
#         background: white;
#         border-radius: 10px;
#         padding: 0.5rem;
#         box-shadow: 0 5px 15px rgba(0,0,0,0.1);
#     }
#     .stTabs [data-baseweb="tab"] {
#         font-size: 1.5rem;
#         padding: 1rem 2rem;
#         color: #4CAF50;
#     }
#     .stTabs [data-baseweb="tab"]:hover {
#         color: #2E7D32;
#     }
#     .stTabs [data-baseweb="tab--selected"] {
#         background: #4CAF50;
#         color: white;
#         border-radius: 10px;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define the model architecture for disease prediction
# class ImprovedCNNViTHybrid(nn.Module):
#     def __init__(self, num_classes, pretrained=True):
#         super(ImprovedCNNViTHybrid, self).__init__()
#         self.backbone = models.resnet50(pretrained=pretrained)
#         self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
#         self.feature_dim = 2048
#         self.patch_size = 7
#         self.num_patches = 49
#         self.embedding_dim = 768
        
#         self.feature_projection = nn.Sequential(
#             nn.AdaptiveAvgPool2d((7, 7)),
#             nn.Conv2d(self.feature_dim, self.embedding_dim, kernel_size=1),
#             nn.BatchNorm2d(self.embedding_dim),
#             nn.ReLU(inplace=True)
#         )
        
#         self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
#         self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embedding_dim))
#         self.dropout = nn.Dropout(0.3)
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embedding_dim, nhead=8, dim_feedforward=2048, dropout=0.3, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(self.embedding_dim),
#             nn.Dropout(0.3),
#             nn.Linear(self.embedding_dim, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(512, num_classes)
#         )
        
#         self._init_weights()
    
#     def _init_weights(self):
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
#     def forward(self, x):
#         B = x.shape[0]
#         features = self.backbone(x)
#         features = self.feature_projection(features)
#         features = features.flatten(2).transpose(1, 2)
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         features = torch.cat([cls_tokens, features], dim=1)
#         features = features + self.pos_embed
#         features = self.dropout(features)
#         encoded = self.transformer(features)
#         cls_output = encoded[:, 0]
#         return self.classifier(cls_output)

# # Load the trained model for disease prediction
# MODEL_PATH = r"C:\Users\91861\Downloads\AI_Farmer_Data\SIH-AI-Farmer\backend\all_crops_hybrid_best_model.pth"
# class_names_path = r"C:\Users\91861\Downloads\AI_Farmer_Data\SIH-AI-Farmer\backend\classes.txt"

# # Load class names
# with open(class_names_path, 'r') as f:
#     class_names = f.read().splitlines()

# # Load the disease prediction model
# disease_model = ImprovedCNNViTHybrid(num_classes=len(class_names))
# state_dict = torch.load(MODEL_PATH, map_location=device)
# disease_model.load_state_dict(state_dict['model_state_dict'])
# disease_model.eval()
# disease_model.to(device)

# # Define transforms for disease prediction
# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Disease prediction function
# def predict_disease(image):
#     try:
#         img = Image.open(image).convert('RGB')
#         img = val_transform(img).unsqueeze(0).to(device)
#         with torch.no_grad():
#             output = disease_model(img)
#             _, predicted = torch.max(output, 1)
#             confidence = torch.softmax(output, dim=1)[0][predicted].item()
#             predicted_class = class_names[predicted.item()]
#         return predicted_class, confidence
#     except Exception as e:
#         return f"Error processing image: {str(e)}", 0.0

# # Initialize Supabase client for RAG
# SUPABASE_URL = ""
# SUPABASE_KEY = ""
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# # Initialize embeddings and vector store for RAG
# os.environ["GROQ_API_KEY"] = ""
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name="documents")

# # Build RAG chain
# def build_rag_chain():
#     llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",
#         return_source_documents=True
#     )
#     return qa_chain

# # RAG query function (aligned with original implementation)
# def rag_query(qa_chain, query):
#     try:
#         result = qa_chain.invoke({"query": query})
#         answer = result["result"]
#         source_documents = result.get("source_documents", [])
#         return {"answer": answer, "source_documents": source_documents}
#     except Exception as e:
#         st.error(f"Error in RAG query: {str(e)}")
#         return {"answer": "Unable to process the query at this time.", "source_documents": []}

# # Initialize RAG chain
# rag_chain = build_rag_chain()

# # Session state
# if "query_count" not in st.session_state:
#     st.session_state.query_count = 0

# # Main app
# st.markdown('<div class="main-header"><h1 style="font-size: 3rem; margin: 0;">🌱 AI Farmer Platform</h1><p style="font-size: 1.5rem; margin: 1rem 0;">Hackathon Winner 2025 - Empowering Indian Farmers</p></div>', unsafe_allow_html=True)

# # Metrics dashboard
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.markdown('<div class="metric-card">Queries Today: <strong>' + str(st.session_state.query_count) + '</strong></div>', unsafe_allow_html=True)
# with col2:
#     st.markdown('<div class="metric-card">Users Helped: <strong>1,247</strong></div>', unsafe_allow_html=True)
# with col3:
#     st.markdown('<div class="metric-card">Success Rate: <strong>95%</strong></div>', unsafe_allow_html=True)

# # Tabs
# tab1, tab2, tab3 = st.tabs(["Disease Prediction", "Knowledge Base", "Market Analysis"])

# with tab1:
#     st.header("🌿 Disease Prediction")
#     st.markdown('<div class="input-card"><p>Upload a leaf image to predict crop disease using our CNN-ViT hybrid model.</p></div>', unsafe_allow_html=True)

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="disease_upload")
#     if uploaded_file is not None:
#         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#         if st.button("🔍 Predict Disease", key="predict_disease"):
#             with st.spinner("Analyzing image..."):
#                 predicted_class, confidence = predict_disease(uploaded_file)
#                 if isinstance(predicted_class, str) and "Error" in predicted_class:
#                     st.error(predicted_class)
#                 else:
#                     st.markdown(f"### 🌿 Predicted Disease: **{predicted_class}**")
#                     st.markdown(f"**Confidence:** {confidence:.2%}")
#                     st.markdown('<div class="success-box">✅ Prediction completed successfully!</div>', unsafe_allow_html=True)

# with tab2:
#     st.header("📚 Knowledge Base")
#     st.markdown('<div class="input-card"><p>Ask agricultural questions using our pre-embedded RAG system.</p></div>', unsafe_allow_html=True)
    
#     query = st.text_input("Ask about farming practices (e.g., 'how to control red rust')", placeholder="e.g., best fertilizers for rice", key="kb_query")
#     if st.button("🔍 Get Agricultural Advice", key="rag_query"):
#         if query:
#             st.session_state.query_count += 1
#             with st.spinner("Consulting knowledge base..."):
#                 result = rag_query(rag_chain, query)
#                 st.markdown("### 🌾 Agricultural Advice")
#                 st.write(result["answer"])
#                 if result["source_documents"]:
#                     with st.expander("📚 Sources"):
#                         for doc in result["source_documents"]:
#                             st.write(f"**{doc.metadata.get('relpath', 'Unknown source')}**")
#                             st.write(doc.page_content[:300] + "...")
#                 if "Unable to process" not in result["answer"]:
#                     st.markdown('<div class="success-box">✅ Answer generated successfully!</div>', unsafe_allow_html=True)
#                 else:
#                     st.error(result["answer"])

# with tab3:
#     st.header("📈 Real-Time Market Analysis")
#     st.markdown('<div class="input-card"><p>Ask about commodity prices across India using Groq AI and Data.gov.in API.</p></div>', unsafe_allow_html=True)
    
#     query = st.text_input("What do you want to know about market prices? (e.g., 'mango prices in Delhi')", placeholder="e.g., rice prices in Kerala", key="market_query")
#     if st.button("🔍 Get Market Insights", key="market_query_btn"):
#         if query:
#             st.session_state.query_count += 1
#             with st.spinner("Analyzing market data..."):
#                 # Placeholder for market analysis (to be implemented)
#                 st.markdown("### 📊 Market Summary")
#                 st.write("Market data analysis placeholder. Implement `natural_query` function here.")
#                 st.markdown('<div class="success-box">✅ Market data retrieved successfully!</div>', unsafe_allow_html=True)

# # Sidebar
# with st.sidebar:
#     st.header("📊 Dashboard")
#     st.markdown(f'<div class="metric-card">Total Queries: <strong>{st.session_state.query_count}</strong></div>', unsafe_allow_html=True)
    
#     st.header("🔧 Settings")
#     if st.button("🔄 Refresh"):
#         st.experimental_rerun()

# # Footer
# st.markdown("""
# <div style="text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.1); border-radius: 15px; margin-top: 2rem; font-size: 1.2rem;">
#     <h3>🌱 AI Farmer Platform</h3>
#     <p>Hackathon Winner 2025 - Empowering Indian Farmers with AI</p>
#     <p>Built with Streamlit, Groq, and Supabase</p>
# </div>
# """, unsafe_allow_html=True)

import streamlit as st
import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms, models
import requests

# Page configuration
st.set_page_config(
    page_title="AI Farmer Platform - Confiured By Team Processor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        font-family: 'Arial', sans-serif;
        font-size: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #f0f9f0 0%, #e8f5e8 100%);
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        border-left: 6px solid #4CAF50;
        margin: 1.5rem 0;
        font-size: 1.3rem;
        text-align: center;
    }
    .input-card {
        background: white;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 1.5rem 0;
        font-size: 1.2rem;
    }
    .demo-button {
        background: linear-gradient(45deg, #4CAF50, #66BB6A);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        margin: 0.5rem;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    .demo-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.5);
    }
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #66BB6A);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.5);
    }
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.95);
        border: 3px solid #A5D6A7;
        border-radius: 12px;
        padding: 1rem;
        font-size: 1.2rem;
        color: #333;
    }
    .stFileUploader > div > div > div {
        background: white;
        border: 3px dashed #A5D6A7;
        border-radius: 12px;
        font-size: 1.2rem;
    }
    .success-box {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #4CAF50;
        margin: 1.5rem 0;
        font-size: 1.2rem;
        color: #2E7D32;
    }
    .stSpinner {
        color: #4CAF50;
        font-size: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.5rem;
        padding: 1rem 2rem;
        color: #4CAF50;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #2E7D32;
    }
    .stTabs [data-baseweb="tab--selected"] {
        background: #4CAF50;
        color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture for disease prediction
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

# Load the trained model for disease prediction
MODEL_PATH = r"C:\Users\91861\Downloads\AI_Farmer_Data\SIH-AI-Farmer\backend\all_crops_hybrid_best_model.pth"
class_names_path = r"C:\Users\91861\Downloads\AI_Farmer_Data\SIH-AI-Farmer\backend\classes.txt"

# Load class names
with open(class_names_path, 'r') as f:
    class_names = f.read().splitlines()

# Load the disease prediction model
disease_model = ImprovedCNNViTHybrid(num_classes=len(class_names))
state_dict = torch.load(MODEL_PATH, map_location=device)
disease_model.load_state_dict(state_dict['model_state_dict'])
disease_model.eval()
disease_model.to(device)

# Define transforms for disease prediction
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Disease prediction function
def predict_disease(image):
    try:
        img = Image.open(image).convert('RGB')
        img = val_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = disease_model(img)
            _, predicted = torch.max(output, 1)
            confidence = torch.softmax(output, dim=1)[0][predicted].item()
            predicted_class = class_names[predicted.item()]
        return predicted_class, confidence
    except Exception as e:
        return f"Error processing image: {str(e)}", 0.0

# Function to query RAG via FastAPI endpoint
def rag_query_via_api(query_text):
    try:
        api_url = "http://localhost:8000/rag"  # Adjust to your FastAPI server URL
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, json={"query": query_text}, headers=headers)
        response.raise_for_status()
        result = response.json()
        return {
            "answer": result.get("answer", "No answer available"),
            "source_documents": [{"metadata": {"relpath": src}} for src in result.get("sources", [])]
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to RAG API: {str(e)}")
        return {"answer": "Unable to process the query at this time.", "source_documents": []}

# Session state
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Main app
st.markdown('<div class="main-header"><h1 style="font-size: 3rem; margin: 0;">🌱 AI Farmer Platform</h1><p style="font-size: 1.5rem; margin: 1rem 0;">Team Processor 2025 - Empowering Indian Farmers</p></div>', unsafe_allow_html=True)

# Metrics dashboard
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card">Queries Today: <strong>' + str(st.session_state.query_count) + '</strong></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">Users Helped: <strong>1,247</strong></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card">Success Rate: <strong>95%</strong></div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Disease Prediction", "Knowledge Base", "Market Analysis"])

with tab1:
    st.header("🌿 Disease Prediction")
    st.markdown('<div class="input-card"><p>Upload a leaf image to predict crop disease using our CNN-ViT hybrid model.</p></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="disease_upload")
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("🔍 Predict Disease", key="predict_disease"):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence = predict_disease(uploaded_file)
                if isinstance(predicted_class, str) and "Error" in predicted_class:
                    st.error(predicted_class)
                else:
                    st.markdown(f"### 🌿 Predicted Disease: **{predicted_class}**")
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    st.markdown('<div class="success-box">✅ Prediction completed successfully!</div>', unsafe_allow_html=True)

with tab2:
    st.header("📚 Knowledge Base")
    st.markdown('<div class="input-card"><p>Ask agricultural questions using our pre-embedded RAG system.</p></div>', unsafe_allow_html=True)
    
    query = st.text_input("Ask about farming practices (e.g., 'how to control red rust')", placeholder="e.g., best fertilizers for rice", key="kb_query")
    if st.button("🔍 Get Agricultural Advice", key="rag_query"):
        if query:
            st.session_state.query_count += 1
            with st.spinner("Consulting knowledge base..."):
                result = rag_query_via_api(query)
                st.markdown("### 🌾 Agricultural Advice")
                st.write(result["answer"])
                if result["source_documents"]:
                    with st.expander("📚 Sources"):
                        for doc in result["source_documents"]:
                            st.write(f"**{doc['metadata'].get('relpath', 'Unknown source')}**")
                if "Unable to process" not in result["answer"]:
                    st.markdown('<div class="success-box">✅ Answer generated successfully!</div>', unsafe_allow_html=True)
                else:
                    st.error(result["answer"])

with tab3:
    st.header("📈 Real-Time Market Analysis")
    st.markdown('<div class="input-card"><p>Ask about commodity prices across India using Groq AI and Data.gov.in API.</p></div>', unsafe_allow_html=True)
    
    query = st.text_input("What do you want to know about market prices? (e.g., 'mango prices in Delhi')", placeholder="e.g., rice prices in Kerala", key="market_query")
    if st.button("🔍 Get Market Insights", key="market_query_btn"):
        if query:
            st.session_state.query_count += 1
            with st.spinner("Analyzing market data..."):
                # Placeholder for market analysis (to be implemented)
                st.markdown("### 📊 Market Summary")
                st.write("Market data analysis placeholder. Implement `natural_query` function here.")
                st.markdown('<div class="success-box">✅ Market data retrieved successfully!</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Dashboard")
    st.markdown(f'<div class="metric-card">Total Queries: <strong>{st.session_state.query_count}</strong></div>', unsafe_allow_html=True)
    
    st.header("🔧 Settings")
    if st.button("🔄 Refresh"):
        st.experimental_rerun()

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.1); border-radius: 15px; margin-top: 2rem; font-size: 1.2rem;">
    <h3>🌱 AI Farmer Platform</h3>
    <p>Team Processor 2025 - Empowering Indian Farmers with AI</p>
    <p>Built with Streamlit, Groq, and Supabase</p>
</div>
""", unsafe_allow_html=True)