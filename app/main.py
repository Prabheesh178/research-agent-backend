import os
import uuid
from typing import List, Optional
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.database import (
    get_db_connection,
    get_user_profile,
    save_user_profile,
    delete_user_data,
    DEFAULT_PROFILE
)
from app.utils.pdf_parser import extract_pdf_chunks
from app.utils.vector_store import add_document_chunks
from app.utils.data_parser import parse_uploaded_data
from app.agents import run_agent_pipeline

app = FastAPI(title="Research Agent Backend API")

# Enable CORS for frontend integration
allowed_origins = ["*"]
if os.getenv("ENV") == "production":
    vercel_url = os.getenv("ALLOWED_ORIGIN", "https://your-vercel-frontend.vercel.app")
    allowed_origins = [vercel_url]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    """
    Render cold start warmup endpoint. Returns awake status with zero system info.
    """
    return {"status": "awake"}

# Request/Response schemas
class ProfileUpdate(BaseModel):
    vocab_level: str
    citation_style: str
    avg_sentence_length: int
    connectors: List[str]
    domain: str
    writing_style: str
    sentence_variety: str
    writing_quirks: Optional[str] = ""

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Backend is running"}

# Uploads path for data files
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")

@app.post("/api/query")
async def query_endpoint(
    prompt: str = Form(...),
    user_id: str = Form(...),
    openai_key: Optional[str] = Form(None),
    tavily_key: Optional[str] = Form(None),
    llm_base_url: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
    files: List[UploadFile] = File([])
):
    """
    Main pipeline entry point. Ingests prompt, optional user keys, and files.
    Validates credentials, intercepts API key disclosures, processes spreadsheets in-memory,
    handles errors cleanly without stack trace leakage, and tracks request timing.
    """
    import time
    request_start_time = time.time()
    
    # 1. API Key presence check
    cleaned_openai_key = openai_key.strip() if openai_key else ""
    cleaned_tavily_key = tavily_key.strip() if tavily_key else ""
    cleaned_base_url = llm_base_url.strip() if llm_base_url else ""
    cleaned_model = llm_model.strip() if llm_model else ""
    
    if not cleaned_openai_key:
        raise HTTPException(
            status_code=400,
            detail="Please provide an API key to use this assistant. You can paste your OpenAI, OpenRouter, or use your local Ollama instance."
        )

    # 2. Intercept API key queries
    prompt_lower = prompt.lower().strip()
    key_queries = [
        "what is my api key", "show me my key", "repeat my key",
        "what's my api key", "reveal my api key", "give me my key",
        "what is my key", "show my api key", "display my api key"
    ]
    if any(kq in prompt_lower for kq in key_queries) or ("key" in prompt_lower and ("show" in prompt_lower or "reveal" in prompt_lower or "repeat" in prompt_lower or "what is" in prompt_lower)):
        return {
            "answer": "Your API key is never stored or accessible. It is used only for this session and discarded after.",
            "references": [],
            "rag_chunks": [],
            "trace_logs": [{"agent": "Security Monitor", "status": "completed", "message": "API key retrieval attempt blocked."}],
            "memory_profile": {},
            "ingested_files": []
        }

    # 3. Process and ingest uploaded files (PDFs to vector search, spreadsheets parsed in-memory only)
    ingested_docs = []
    uploaded_data_sheets = []
    if files:
        for file in files:
            fn = file.filename.lower()
            
            # Case A: PDF ingestion
            if fn.endswith(".pdf"):
                try:
                    content = await file.read()
                    doc_id = str(uuid.uuid4())
                    
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO documents (id, user_id, filename) VALUES (?, ?, ?)",
                        (doc_id, user_id, file.filename)
                    )
                    conn.commit()
                    conn.close()
                    
                    chunks = extract_pdf_chunks(content, file.filename)
                    add_document_chunks(doc_id, user_id, chunks, cleaned_openai_key, cleaned_base_url)
                    ingested_docs.append(file.filename)
                except Exception as e:
                    print(f"Error ingesting PDF {file.filename}: {str(e)}")
                    # Clean user-facing error message only
                    raise HTTPException(status_code=500, detail=f"Failed to ingest PDF '{file.filename}'. Please try again.")
            
            # Case B: Tabular data file ingestion (processed in memory, never written to disk)
            elif fn.endswith((".csv", ".tsv", ".json", ".xlsx", ".xls")):
                try:
                    content = await file.read()
                    doc_id = str(uuid.uuid4())
                    
                    # Parse in memory immediately
                    parsed_sheet = parse_uploaded_data(content, file.filename)
                    uploaded_data_sheets.append({
                        "filename": file.filename,
                        "parsed_data": parsed_sheet
                    })
                    
                    # Record metadata in SQLite so document list shows the file
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO documents (id, user_id, filename) VALUES (?, ?, ?)",
                        (doc_id, user_id, file.filename)
                    )
                    conn.commit()
                    conn.close()
                    
                    ingested_docs.append(file.filename)
                except Exception as e:
                    print(f"Error parsing data file {file.filename}: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Failed to process data file '{file.filename}'. Please check formatting.")
                
    # 4. Run the Multi-Agent Pipeline
    try:
        result_state = run_agent_pipeline(
            prompt=prompt,
            user_id=user_id,
            openai_key=cleaned_openai_key,
            tavily_key=cleaned_tavily_key,
            llm_base_url=cleaned_base_url,
            llm_model=cleaned_model,
            uploaded_data_sheets=uploaded_data_sheets
        )
    except Exception as e:
        # Enforce clean error messaging
        err_msg = str(e)
        if "api_key" in err_msg.lower() or "sk-" in err_msg or "token" in err_msg.lower():
            err_msg = "The API key provided appears to be invalid or has exceeded its quota. Please check your key and try again."
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {err_msg}")
        
    return {
        "answer": result_state["final_output"],
        "references": result_state["web_papers"],
        "rag_chunks": [
            {
                "filename": chunk["filename"],
                "page": chunk["page_num"],
                "text": chunk["text"][:200] + "..."
            }
            for chunk in result_state.get("rag_chunks", [])
        ],
        "trace_logs": result_state["trace_logs"],
        "memory_profile": result_state["memory_profile"],
        "ingested_files": ingested_docs
    }

@app.get("/api/profile/{user_id}")
def get_profile_endpoint(user_id: str):
    profile = get_user_profile(user_id)
    if not profile:
        profile = DEFAULT_PROFILE.copy()
    return {"profile": profile}

@app.put("/api/profile/{user_id}")
def update_profile_endpoint(user_id: str, update: ProfileUpdate):
    profile = get_user_profile(user_id)
    if not profile:
        profile = DEFAULT_PROFILE.copy()
        
    # Update styling fields
    profile["vocab_level"] = update.vocab_level
    profile["citation_style"] = update.citation_style
    profile["avg_sentence_length"] = update.avg_sentence_length
    profile["connectors"] = update.connectors
    profile["domain"] = update.domain
    profile["writing_style"] = update.writing_style
    profile["sentence_variety"] = update.sentence_variety
    profile["writing_quirks"] = update.writing_quirks
    
    save_user_profile(user_id, profile)
    return {"status": "success", "profile": profile}

@app.get("/api/documents/{user_id}")
def get_documents_endpoint(user_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, upload_time FROM documents WHERE user_id = ? ORDER BY upload_time DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    docs = [{"id": r["id"], "filename": r["filename"], "upload_time": r["upload_time"]} for r in rows]
    return {"documents": docs}

@app.delete("/api/documents/{user_id}/{doc_id}")
def delete_document_endpoint(user_id: str, doc_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    # verify ownership and get filename
    cursor.execute("SELECT filename FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found or access denied")
        
    filename = row["filename"]
    
    # delete chunks and doc metadata
    cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    
    # delete local file if it's a saved data file
    try:
        local_path = os.path.join(UPLOADS_DIR, user_id, filename)
        if os.path.exists(local_path):
            os.remove(local_path)
    except Exception as e:
        print(f"Error removing local file {filename}: {str(e)}")
        
    return {"status": "success", "message": "Document deleted successfully"}

@app.delete("/api/profile/{user_id}")
def clear_user_data_endpoint(user_id: str):
    delete_user_data(user_id)
    
    # clear uploads directory
    try:
        user_upload_dir = os.path.join(UPLOADS_DIR, user_id)
        if os.path.exists(user_upload_dir):
            import shutil
            shutil.rmtree(user_upload_dir)
    except Exception as e:
        print(f"Error clearing user uploads folder: {str(e)}")
        
    return {"status": "success", "message": f"All data for user {user_id} has been wiped."}
