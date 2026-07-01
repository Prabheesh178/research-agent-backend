from app.utils.vector_store import get_openai_client

def run_humanizer_agent(state: dict) -> dict:
    """
    Runs on all outputs. Humanizer Agent reads synthesis or paper writer output
    and adapts it to sound like the user (perplexity, burstiness, vocab, connectors, writing_quirks).
    """
    state["trace_logs"].append({
        "agent": "Humanizer",
        "status": "running",
        "message": "Adjusting perplexity, sentence lengths, and style matching user profile..."
    })
    
    intent = state["intent"]
    openai_key = state["openai_key"]
    memory_profile = state["memory_profile"]
    
    # Select which text to humanize
    if intent == "QA":
        text_to_humanize = state.get("synthesis_output", "")
    elif intent == "PAPER":
        text_to_humanize = state.get("paper_writer_output", "")
    else:  # BOTH
        # Join both QA response and paper
        qa = state.get("synthesis_output", "")
        paper = state.get("paper_writer_output", "")
        text_to_humanize = f"{qa}\n\n=========================================\n\n{paper}"
        
    if not text_to_humanize or text_to_humanize.strip() == "":
        state["final_output"] = "Error: No draft text generated to humanize."
        state["trace_logs"].append({
            "agent": "Humanizer",
            "status": "error",
            "message": "No text content found to process."
        })
        return state
        
    client = get_openai_client(openai_key, state.get("llm_base_url"))
    
    humanizer_system_prompt = (
        "You are an academic writing Humanizer. Your job is to rewrite academic drafts to match a user's writing fingerprint. "
        "You must improve perplexity and burstiness according to these specifications:\n\n"
        "1. PERPLEXITY (Word Choice):\n"
        "   - Remove AI clichés: 'It is worth noting that' (delete), 'Leveraging' (use 'using'/'applying'), 'In the realm of' (use 'in'/'within'), 'This study aims to' (use 'this study investigates'/'examines').\n"
        "   - Limit repetition of 'significantly' or 'furthermore'.\n"
        "   - Use these custom connectors naturally: " + ", ".join(memory_profile.get("connectors", [])) + ".\n\n"
        "2. BURSTINESS (Sentence Variety):\n"
        "   - Vary sentence lengths: mix short, punchy statements (under 10 words) with longer compound sentences.\n"
        "   - Aim for an average sentence length of " + str(memory_profile.get("avg_sentence_length", 20)) + " words.\n"
        "   - Add a short sentence after every 3-4 long sentences.\n\n"
        "3. STYLE & VOCAB:\n"
        "   - Match vocabulary level: " + memory_profile.get("vocab_level", "postgraduate") + ".\n"
        "   - Match writing style: " + memory_profile.get("writing_style", "formal") + ".\n"
        "   - Apply these writing quirks: " + memory_profile.get("writing_quirks", "None") + ".\n\n"
        "CRITICAL RULES:\n"
        "- DO NOT alter any inline citations (e.g. [1], [PDF: file.pdf, p.3]) or reference names/links.\n"
        "- DO NOT alter technical formulas, numerical data, headers, or bibliography lists.\n"
        "- Maintain third-person academic tone unless user writing quirks state otherwise."
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o",  # Quality is essential for style transfer
            messages=[
                {"role": "system", "content": humanizer_system_prompt},
                {"role": "user", "content": f"Please humanize this draft:\n\n{text_to_humanize}"}
            ],
            temperature=0.4
        )
        humanized_text = response.choices[0].message.content.strip()
    except Exception as e:
        humanized_text = f"Failed to humanize draft: {str(e)}\n\n--- Raw Draft ---\n\n{text_to_humanize}"
        state["trace_logs"].append({
            "agent": "Humanizer",
            "status": "warning",
            "message": f"Humanizer API call failed: {str(e)}. Using raw draft instead."
        })
        
    state["final_output"] = humanized_text
    
    state["trace_logs"].append({
        "agent": "Humanizer",
        "status": "completed",
        "message": "Humanizer process completed successfully."
    })
    
    return state
