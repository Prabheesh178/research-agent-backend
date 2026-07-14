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
    model = state.get("llm_model") or "gpt-4o"
    
    client = get_openai_client(openai_key, state.get("llm_base_url"))
    
    # Format tabular data if any (fixed missing data_context)
    data_context = ""
    uploaded_data_sheets = state.get("uploaded_data_sheets", [])
    if uploaded_data_sheets:
        data_context += "--- LOCAL TABULAR SHEET DATA (Insights / Numerical context) ---\n"
        for sheet in uploaded_data_sheets:
            data_context += f"Filename: {sheet['filename']}\n"
            data_context += f"Data Snippet/Parsed Results: {str(sheet.get('parsed_data', ''))[:1000]}\n\n"
            
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
    
    # Check if a full paper is requested
    is_full_paper = any(kw in prompt_lower for kw in ["full paper", "complete paper", "all sections", "write a paper", "generate a paper"])
    
    # Check for prompt extensions from active skills
    skill_extension = state.get("system_prompt_extension", "")
    
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
    if skill_extension:
        user_content += f"\n\n--- ACTIVE SKILL CUSTOM INSTRUCTIONS ---\n{skill_extension}\n"

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
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3
            )
            paper_text = response.choices[0].message.content.strip()
        except Exception as e:
            paper_text = f"Error during paper continuation: {str(e)}"
            state["trace_logs"].append({
                "agent": "Paper Writer Agent",
                "status": "error",
                "message": f"Continuation failed: {str(e)}"
            })
            
    elif is_full_paper:
        paper_text = ""
        # Multi-pass generators (merged 3 calls vs paid 6 calls)
        if state.get("merge_paper_sections"):
            # 3 Merged Calls (Compression 5)
            state["trace_logs"].append({
                "agent": "Paper Writer Agent",
                "status": "info",
                "message": "Free/Local budget optimization: Merging sections into 3 calls."
            })
            
            # Call A: Title + Abstract + Intro
            try:
                state["trace_logs"].append({
                    "agent": "Paper Writer Agent",
                    "status": "running",
                    "message": "Generating Title, Abstract, and Introduction..."
                })
                sys_prompt = "You are an academic Paper Writer Agent. Write the TITLE, ABSTRACT (150-250 words, no citations), and INTRODUCTION sections."
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.3
                )
                paper_text += response.choices[0].message.content.strip() + "\n\n"
            except Exception as e:
                paper_text += f"Error during Title/Abstract/Intro generation: {str(e)}\n\n"
                
            # Call B: Lit Review + Methodology
            try:
                state["trace_logs"].append({
                    "agent": "Paper Writer Agent",
                    "status": "running",
                    "message": "Generating Literature Review and Methodology..."
                })
                sys_prompt = (
                    f"You are an academic Paper Writer Agent. Write the LITERATURE REVIEW and METHODOLOGY sections. "
                    f"Group papers into thematic clusters and cite them. Connect seamlessly to previous draft:\n\n{paper_text}"
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.3
                )
                paper_text += response.choices[0].message.content.strip() + "\n\n"
            except Exception as e:
                paper_text += f"Error during Lit Review/Methodology generation: {str(e)}\n\n"
                
            # Call C: Results + Conclusion + References
            try:
                state["trace_logs"].append({
                    "agent": "Paper Writer Agent",
                    "status": "running",
                    "message": "Generating Results, Discussion, Conclusion, and References..."
                })
                sys_prompt = (
                    f"You are an academic Paper Writer Agent. Write the RESULTS & DISCUSSION, CONCLUSION, and REFERENCES sections. "
                    f"Connect seamlessly to previous draft:\n\n{paper_text}"
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.3
                )
                paper_text += response.choices[0].message.content.strip()
            except Exception as e:
                paper_text += f"Error during Results/Conclusion generation: {str(e)}\n\n"
        else:
            # 6 Separate Calls (Paid Tier)
            state["trace_logs"].append({
                "agent": "Paper Writer Agent",
                "status": "info",
                "message": "Paid tier active: Generating full paper in 6 sequential section calls."
            })
            sections = [
                ("Title & Abstract", "Write the TITLE and ABSTRACT (150-250 words, no citations)."),
                ("Introduction", "Write the INTRODUCTION section (background, gap, and structure)."),
                ("Literature Review", "Write the LITERATURE REVIEW section. Group papers into thematic clusters and cite them."),
                ("Methodology", "Write the METHODOLOGY section (data collection, tools, and protocols)."),
                ("Results & Discussion", "Write the RESULTS & DISCUSSION section detailing findings."),
                ("Conclusion & References", "Write the CONCLUSION and REFERENCES sections.")
            ]
            
            for sec_name, sec_prompt in sections:
                try:
                    state["trace_logs"].append({
                        "agent": "Paper Writer Agent",
                        "status": "running",
                        "message": f"Generating {sec_name}..."
                    })
                    sys_prompt = (
                        f"You are an academic Paper Writer Agent. {sec_prompt}\n"
                        f"Connect it seamlessly to the previously written sections:\n\n{paper_text}" if paper_text else f"You are an academic Paper Writer Agent. {sec_prompt}"
                    )
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.3
                    )
                    paper_text += response.choices[0].message.content.strip() + "\n\n"
                except Exception as e:
                    paper_text += f"Error during {sec_name} generation: {str(e)}\n\n"
            paper_text = paper_text.strip()
            
    else:
        # Standard Single-pass (for specific sections or shorter prompts)
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
            "If the user asks for multiple long sections, generate only the first 2-3 sections, and append: "
            "'\n\nSection complete. Type 'continue' for the next section.' so that the request does not timeout on the server."
        )
        try:
            response = client.chat.completions.create(
                model=model,
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
