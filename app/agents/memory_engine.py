import json
from app.database import save_user_profile, get_user_profile
from app.utils.vector_store import get_openai_client

def run_memory_engine(state: dict) -> dict:
    """
    Runs silently after response delivery. Memory Engine analyzes user prompt and
    final output, extracts style attributes, and merges them 30/70 with existing profile.
    """
    state["trace_logs"].append({
        "agent": "Memory Engine",
        "status": "running",
        "message": "Analyzing writing footprint and updating user style profile..."
    })
    
    user_id = state["user_id"]
    prompt = state["prompt"]
    final_output = state["final_output"]
    openai_key = state["openai_key"]
    existing_profile = state["memory_profile"]
    
    client = get_openai_client(openai_key, state.get("llm_base_url"))
    
    # We ask the LLM to extract the current session metrics and merge with existing
    merge_prompt = (
        "You are a Style Memory Engine. Analyze this research session and update the user style profile.\n\n"
        "Session Details:\n"
        f"- User's Input Prompt: \"{prompt}\"\n"
        f"- Generated Response Snippet: \"{final_output[:600]}...\"\n\n"
        "Existing Profile JSON:\n"
        f"{json.dumps(existing_profile, indent=2)}\n\n"
        "Instructions:\n"
        "1. Extract new style observations from the user's prompt:\n"
        "   - domain (e.g. computer science, medicine, etc.)\n"
        "   - citation_style (e.g. IEEE, APA, Harvard)\n"
        "   - vocab_level (e.g. graduate, postgraduate, undergraduate)\n"
        "   - avg_sentence_length (approximate words per sentence in user input)\n"
        "   - connectors (connective words they used in their prompt)\n"
        "   - writing_style (formal / semi-formal / technical)\n"
        "   - sentence_variety (high / moderate / low)\n"
        "   - writing_quirks (distinct patterns, typos, short-hands, preference for formatting)\n"
        "2. Merge observations with the existing profile using a 30% new/70% existing weight:\n"
        "   - Numerical values like avg_sentence_length: new_avg = (existing_avg * 0.7) + (session_avg * 0.3)\n"
        "   - Text categories: only change if session indicates a persistent shift\n"
        "   - Connectors: merge and keep top 6-8 unique connectors\n"
        "   - topics_researched: append the core topic of this session to the list (keep maximum 20 items, unique)\n"
        "   - Increment session_count by 1.\n\n"
        "Respond ONLY with the raw updated JSON object containing the profile."
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a style parser that outputs raw JSON."},
                {"role": "user", "content": merge_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        updated_profile = json.loads(response.choices[0].message.content)
        
        # Ensure session count actually increments
        updated_profile["session_count"] = existing_profile.get("session_count", 0) + 1
        
        # Persist, append, or preserve the last generated draft
        if state.get("paper_writer_output"):
            current_draft = state["paper_writer_output"]
            prompt_lower = state["prompt"].lower().strip()
            is_continue = prompt_lower in ["continue", "next", "continue writing", "proceed", "next section"]
            if is_continue and existing_profile.get("last_draft"):
                updated_profile["last_draft"] = existing_profile["last_draft"] + "\n\n" + current_draft
            else:
                updated_profile["last_draft"] = current_draft
        elif existing_profile.get("last_draft"):
            updated_profile["last_draft"] = existing_profile["last_draft"]
            
        # Save to DB
        save_user_profile(user_id, updated_profile)
        state["memory_profile"] = updated_profile
        
        state["trace_logs"].append({
            "agent": "Memory Engine",
            "status": "completed",
            "message": "Memory profile updated successfully.",
            "data": {
                "updated_profile": updated_profile
            }
        })
    except Exception as e:
        state["trace_logs"].append({
            "agent": "Memory Engine",
            "status": "error",
            "message": f"Memory engine failed to merge: {str(e)}"
        })
        
    return state
