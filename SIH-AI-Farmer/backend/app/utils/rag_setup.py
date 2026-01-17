from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from pathlib import Path
import os
import pandas as pd
from app.lib.supabase import create_supabase_client

# Custom loader for Excel files
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

# Generic file loader
def load_file_to_documents(file_path: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(file_path).load()
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

# Recursive document gatherer
def gather_documents_recursive(root_dir: str, allowed_ext=None) -> list[Document]:
    if allowed_ext is None:
        allowed_ext = {".pdf", ".txt", ".csv", ".xls", ".xlsx", ".docx"}
    all_docs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if Path(name).suffix.lower() in allowed_ext:
                fpath = os.path.join(dirpath, name)
                docs = load_file_to_documents(fpath)
                for d in docs:
                    d.metadata["relpath"] = os.path.relpath(fpath, root_dir)
                    d.metadata["subfolder"] = os.path.relpath(dirpath, root_dir)
                all_docs.extend(docs)
    return all_docs

# Build RAG chain (called once during setup)
def build_rag_chain(root_dir: str, groq_api_key: str, model="llama-3.3-70b-versatile"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"[INFO] Loading documents from {root_dir} ...")
    docs = gather_documents_recursive(root_dir)
    print(f"[INFO] Loaded {len(docs)} raw documents. Splitting ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks. Building Supabase vector store ...")
    vectorstore = SupabaseVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=create_supabase_client(),
        table_name="documents"
    )
    llm = ChatGroq(api_key=groq_api_key, model=model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain

# Query function for runtime use
def rag_query(query: str, qa_chain):
    """Execute a query using the pre-built RAG chain."""
    try:
        result = qa_chain({"query": query})
        return {
            "answer": result["result"],
            "source_documents": [{"content": doc.page_content, "metadata": doc.metadata} for doc in result["source_documents"]]
        }
    except Exception as e:
        raise RuntimeError(f"RAG query failed: {str(e)}")