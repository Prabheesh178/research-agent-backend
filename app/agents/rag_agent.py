from app.utils.vector_store import search_similarity, get_openai_client

def run_rag_agent(state: dict) -> dict:
    """
    Step 1: Extract search query from prompt (optional but helpful)
    Step 2: Vector search on uploaded documents
    Step 3: Generate a 2-3 sentence internal summary of findings
    """
    state["trace_logs"].append({
        "agent": "RAG Agent",
        "status": "running",
        "message": "Searching uploaded documents for context..."
    })
    
    user_id = state["user_id"]
    prompt = state["prompt"]
    openai_key = state["openai_key"]
    base_url = state.get("llm_base_url")
    model = state.get("llm_model") or "gpt-4o-mini"
    
    # 1. Generate clean search query from prompt
    client = get_openai_client(openai_key, base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a research assistant. Extract the academic topic or question from the user's prompt as a clean search query (max 10 words). Output ONLY the search query."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        search_query = response.choices[0].message.content.strip().strip('"')
    except Exception:
        search_query = prompt  # Fallback
        
    state["trace_logs"].append({
        "agent": "RAG Agent",
        "status": "info",
        "message": f"Generated vector search query: '{search_query}'"
    })
    
    # 2. Perform SQLite Similarity Search
    try:
        results = search_similarity(user_id, search_query, openai_key, top_k=5, base_url=base_url)
    except Exception as e:
        results = []
        state["trace_logs"].append({
            "agent": "RAG Agent",
            "status": "error",
            "message": f"Vector search failed: {str(e)}"
        })
        
    state["rag_chunks"] = results
    
    if not results:
        # Check if they have PDF documents uploaded in the SQLite documents metadata table
        has_docs = False
        try:
            from app.database import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM documents WHERE user_id = ? AND filename LIKE '%.pdf'", (user_id,))
            pdf_count = cursor.fetchone()["count"]
            conn.close()
            has_docs = pdf_count > 0
        except Exception:
            pass

        if has_docs:
            state["final_output"] = "It looks like the session expired and your documents need to be re-uploaded. Please upload your PDFs again."
            state["trace_logs"].append({
                "agent": "RAG Agent",
                "status": "error",
                "message": "Session expired. Document chunks are missing from the vector store."
            })
            raise ValueError(state["final_output"])

        state["rag_summary"] = "No relevant content in uploaded documents"
        state["trace_logs"].append({
            "agent": "RAG Agent",
            "status": "completed",
            "message": "No relevant local documents found.",
            "data": {"summary": state["rag_summary"], "chunks": []}
        })
        return state
        
    # Format chunks for summary prompt
    chunks_text = ""
    for r in results:
        chunks_text += f"Source: {r['filename']}, Page: {r['page_num']}\nContent: {r['text']}\n\n"
        
    # 3. Generate 2-3 sentence internal summary using OpenAI
    summary_system_prompt = (
        "You are a academic research RAG summarizer. Read the following document chunks and produce a 2-3 sentence internal summary. "
        "Summarize what the user's documents say about their topic. If the documents are not relevant, state 'No relevant content in uploaded documents'. "
        "Keep the summary factual and dry. Do not write for the end-user, write it as a research summary note for another AI."
    )
    
    try:
        summary_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": summary_system_prompt},
                {"role": "user", "content": f"Topic: {search_query}\n\nChunks:\n{chunks_text}"}
            ],
            temperature=0.2
        )
        rag_summary = summary_response.choices[0].message.content.strip()
    except Exception as e:
        rag_summary = "Failed to generate summary from uploaded documents."
        
    state["rag_summary"] = rag_summary
    
    # Format chunk references for frontend view
    chunks_info = [{"filename": r["filename"], "page": r["page_num"], "snippet": r["text"][:150] + "..."} for r in results]
    
    state["trace_logs"].append({
        "agent": "RAG Agent",
        "status": "completed",
        "message": f"Retrieved {len(results)} relevant chunks and created internal summary.",
        "data": {
            "summary": rag_summary,
            "chunks": chunks_info
        }
    })
    
    return state
