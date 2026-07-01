from app.utils.vector_store import get_openai_client

def run_synthesis_agent(state: dict) -> dict:
    """
    Runs only for QA path. Synthesis Agent merges local RAG context + Web Research.
    Generates academic answer structure: Answer paragraph -> Key Findings -> References list.
    """
    state["trace_logs"].append({
        "agent": "Synthesis Agent",
        "status": "running",
        "message": "Synthesizing research results and drafting answer..."
    })
    
    prompt = state["prompt"]
    openai_key = state["openai_key"]
    memory_profile = state["memory_profile"]
    rag_summary = state["rag_summary"]
    rag_chunks = state.get("rag_chunks", [])
    web_papers = state.get("web_papers", [])
    
    client = get_openai_client(openai_key, state.get("llm_base_url"))
    
    # Format sources for LLM prompt
    rag_context = ""
    if rag_chunks and rag_summary != "No relevant content in uploaded documents":
        rag_context += "--- UPLOADED PDF CONTEXT ---\n"
        for idx, chunk in enumerate(rag_chunks):
            rag_context += f"Citation Label: [PDF: {chunk['filename']}, p.{chunk['page_num']}]\n"
            rag_context += f"Text snippet: {chunk['text']}\n\n"
    else:
        rag_context += "No relevant local documents uploaded.\n\n"
        
    web_context = ""
    if web_papers:
        web_context += "--- ACADEMIC WEB SEARCH PAPERS ---\n"
        for idx, paper in enumerate(web_papers):
            web_context += f"Citation Label: [{idx + 1}]\n"
            web_context += f"Title: {paper['title']}\n"
            web_context += f"Authors: {paper['authors']}\n"
            web_context += f"Year: {paper['year']}\n"
            web_context += f"Venue: {paper['venue']}\n"
            web_context += f"URL/DOI: {paper['url']}\n"
            web_context += f"Abstract: {paper.get('abstract', '')[:300]}\n\n"
    else:
        web_context += "No academic web papers found.\n\n"
        
    system_prompt = (
        "You are an academic Synthesis Agent. Your goal is to write a detailed, cited response to the user's research query using ONLY the provided sources.\n\n"
        "Rules:\n"
        "1. Every factual claim must be backed by a source. Use standard inline citations: [1], [2], or [PDF: filename, p.X].\n"
        "2. If you make a claim that has no source in the retrieved context, you MUST label it: '(model inference — no source retrieved)'. Never invent a source.\n"
        "3. Match the vocabulary level of the user's profile: " + memory_profile.get("vocab_level", "postgraduate") + ".\n"
        "4. Strict Structure:\n"
        "   - ANSWER PARAGRAPH: Write a direct academic answer (1-2 paragraphs) to the user's prompt citing appropriate sources.\n"
        "   - KEY FINDINGS: Bulleted list summarizing core takeaways with citations.\n"
        "   - REFERENCES: A numbered list mapping citations back to the sources. For PDFs, output '[PDF: filename, p.X]'. For web papers, output '[X] Author. Title. Venue. Year. URL/DOI'."
    )
    
    data_context = ""
    data_analysis = state.get("data_analysis_output", "")
    if data_analysis:
        data_context = f"--- STRUCTURED DATA ANALYSIS RESULTS ---\n{data_analysis}\n\n"

    user_content = (
        f"User Prompt: {prompt}\n\n"
        f"Memory Profile Style Guidelines:\n"
        f"- Vocabulary Level: {memory_profile.get('vocab_level', 'postgraduate')}\n"
        f"- Citation Style: {memory_profile.get('citation_style', 'IEEE')}\n\n"
        f"Available Context:\n"
        f"{data_context}"
        f"{rag_context}\n"
        f"{web_context}"
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o",  # Synthesis is critical, use gpt-4o as prompt directs
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error during synthesis processing: {str(e)}"
        state["trace_logs"].append({
            "agent": "Synthesis Agent",
            "status": "error",
            "message": f"Synthesis generation failed: {str(e)}"
        })
        
    state["synthesis_output"] = answer
    
    state["trace_logs"].append({
        "agent": "Synthesis Agent",
        "status": "completed",
        "message": "Research synthesis complete. Cited draft generated."
    })
    
    return state
