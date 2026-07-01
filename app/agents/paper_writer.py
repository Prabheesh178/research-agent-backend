from app.utils.vector_store import get_openai_client

def run_paper_writer_agent(state: dict) -> dict:
    """
    Runs only for PAPER or BOTH paths. Paper Writer Agent structures
    academic sections: Abstract -> Intro -> Lit Review -> Method -> Results -> Conclusion.
    Cites local RAG/Web results using IEEE/APA.
    """
    state["trace_logs"].append({
        "agent": "Paper Writer Agent",
        "status": "running",
        "message": "Generating structured academic paper draft..."
    })
    
    prompt = state["prompt"]
    openai_key = state["openai_key"]
    memory_profile = state["memory_profile"]
    rag_summary = state["rag_summary"]
    rag_chunks = state.get("rag_chunks", [])
    web_papers = state.get("web_papers", [])
    
    client = get_openai_client(openai_key, state.get("llm_base_url"))
    
    # Format resources
    rag_context = ""
    if rag_chunks and rag_summary != "No relevant content in uploaded documents":
        rag_context += "--- LOCAL UPLOADED PDF DATA (Methodology / Results context) ---\n"
        for chunk in rag_chunks:
            rag_context += f"Citation Label: [PDF: {chunk['filename']}, p.{chunk['page_num']}]\n"
            rag_context += f"Snippet: {chunk['text']}\n\n"
            
    web_context = ""
    if web_papers:
        web_context += "--- ACADEMIC RESEARCH SOURCES (Literature Review context) ---\n"
        for idx, paper in enumerate(web_papers):
            web_context += f"Citation Label: [{idx + 1}]\n"
            web_context += f"Title: {paper['title']}\n"
            web_context += f"Authors: {paper['authors']}\n"
            web_context += f"Year: {paper['year']}\n"
            web_context += f"Venue: {paper['venue']}\n"
            web_context += f"URL/DOI: {paper['url']}\n"
            web_context += f"Abstract: {paper.get('abstract', '')}\n\n"
            
    # Detect if we are resuming a previously split draft
    prompt_lower = prompt.lower().strip()
    is_continue = prompt_lower in ["continue", "next", "continue writing", "proceed", "next section"]
    last_draft = memory_profile.get("last_draft", "")
    
    if is_continue and last_draft:
        system_prompt = (
            "You are an academic Paper Writer Agent. The user wants to CONTINUE writing their paper draft.\n"
            "Here is the draft written so far:\n\n"
            "--- START OF PREVIOUS DRAFT ---\n"
            f"{last_draft}\n"
            "--- END OF PREVIOUS DRAFT ---\n\n"
            "Analyze where the previous draft left off. Identify which sections are missing or incomplete. "
            "Generate the next logical sections (e.g. Methodology, Results & Discussion, Conclusion, References). "
            "Do NOT repeat the sections that have already been fully written. Connect your continuation seamlessly. "
            "If the remaining sections are very long, focus on the next 1 or 2 sections, and append: "
            "'\n\nSection complete. Type 'continue' for the next section.' to avoid timeout."
        )
        user_content = (
            f"Please continue writing the research paper from where the previous draft left off.\n\n"
            f"Style Guidelines:\n"
            f"- Vocabulary Level: {memory_profile.get('vocab_level', 'postgraduate')}\n"
            f"- Citation Style: {memory_profile.get('citation_style', 'IEEE')}\n\n"
            f"Available Context:\n"
            f"{rag_context}\n"
            f"{web_context}"
        )
    else:
        system_prompt = (
            "You are an academic Paper Writer Agent. Write the requested paper or section using the retrieved sources.\n\n"
            "Section Order Guidelines (Generate ONLY what the user prompt requests):\n"
            "1. TITLE\n"
            "2. ABSTRACT (150-250 words, no citations)\n"
            "3. INTRODUCTION (background, problem, gap, contributions, structure)\n"
            "4. LITERATURE REVIEW (group papers into 2-4 thematic clusters, end with the gap)\n"
            "5. METHODOLOGY (use user's PDF data here if available)\n"
            "6. RESULTS & DISCUSSION\n"
            "7. CONCLUSION (contributions, limitations, future work)\n"
            "8. REFERENCES\n\n"
            "TIMEOUT PREVENTION RULE:\n"
            "If the user asks for a 'full paper' or multiple long sections, generate only the first 2-3 sections "
            "(e.g., Title, Abstract, Introduction, and optionally Literature Review), and append: "
            "'\n\nSection complete. Type 'continue' for the next section.' so that the request does not timeout on the server."
        )
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
            model=state.get("llm_model") or "gpt-4o",  # Paper writer needs high capabilities, use gpt-4o
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3
        )
        paper_text = response.choices[0].message.content.strip()
    except Exception as e:
        paper_text = f"Error during paper generation: {str(e)}"
        state["trace_logs"].append({
            "agent": "Paper Writer Agent",
            "status": "error",
            "message": f"Paper writer failed: {str(e)}"
        })
        
    state["paper_writer_output"] = paper_text
    
    state["trace_logs"].append({
        "agent": "Paper Writer Agent",
        "status": "completed",
        "message": "Academic paper draft generated."
    })
    
    return state
