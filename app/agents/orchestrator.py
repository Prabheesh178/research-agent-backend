import json
from app.database import get_user_profile, save_user_profile, DEFAULT_PROFILE, get_db_connection
from app.utils.vector_store import get_openai_client

def orchestrate_session(state: dict) -> dict:
    """
    Step 1: Classify intent (QA, PAPER, BOTH)
    Step 2: Load user memory profile
    """
    state["trace_logs"].append({
        "agent": "Orchestrator",
        "status": "running",
        "message": "Initializing session & loading user memory profile..."
    })
    
    # 1. Load memory profile
    user_id = state["user_id"]
    profile = get_user_profile(user_id)
    if not profile:
        profile = DEFAULT_PROFILE.copy()
        save_user_profile(user_id, profile)
        state["trace_logs"].append({
            "agent": "Orchestrator",
            "status": "info",
            "message": "No profile found. Created default postgraduate profile."
        })
    else:
        profile["session_count"] = profile.get("session_count", 0) + 1
        save_user_profile(user_id, profile)
        state["trace_logs"].append({
            "agent": "Orchestrator",
            "status": "info",
            "message": f"Loaded profile for user '{user_id}' (Session #{profile['session_count']})."
        })
    
    state["memory_profile"] = profile
    
    # 2. Query user's uploaded files to inform classification
    uploaded_files = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM documents WHERE user_id = ?", (user_id,))
        uploaded_files = [r["filename"] for r in cursor.fetchall()]
        conn.close()
    except Exception:
        pass
        
    # 3. Classify intent using OpenAI
    client = get_openai_client(state["openai_key"], state.get("llm_base_url"))
    prompt = state["prompt"]
    
    # 3a. Check for 'continue' prompt to resume a draft
    prompt_lower = prompt.lower().strip()
    is_continue = prompt_lower in ["continue", "next", "continue writing", "proceed", "next section"]
    if is_continue and profile.get("last_draft"):
        state["intent"] = "PAPER"
        state["trace_logs"].append({
            "agent": "Orchestrator",
            "status": "info",
            "message": "Continue prompt detected. Pre-loading previous draft context and routing to Paper Writer."
        })
        return state

    files_context = f"Uploaded files in workspace: {', '.join(uploaded_files)}" if uploaded_files else "No files uploaded."
    
    system_prompt = (
        "You are an Orchestrator Agent. Your task is to classify the user's research query into one or more of these intents:\n"
        "- 'QA': User wants an answer, summary, explanation, or search for citations. Keywords: 'what', 'explain', 'summarize', 'find papers', 'compare', 'how does'.\n"
        "- 'PAPER': User wants an academic paper, section, abstract, lit review, or draft written. Keywords: 'write', 'draft', 'generate', 'create a section', 'IEEE format'.\n"
        "- 'BOTH': User wants both QA research AND a draft paper written.\n"
        "- 'DATA': User wants numerical analysis, charts, statistics, formatting, duplicate cleaning, columns comparison, or insights from data sheets. Keywords: 'analyze', 'analyse', 'analysis', 'chart', 'graph', 'plot', 'visualize', 'trend', 'pattern', 'correlation', 'distribution', 'average', 'mean', 'median', 'sum', 'count', 'max', 'min', 'duplicates', 'missing values'.\n"
        "- 'DATA+QA': User has uploaded data files (CSV, Excel, JSON) and wants an analysis QA response.\n"
        "- 'DATA+PAPER': User wants to analyze data first, and then write an academic paper section based on it.\n\n"
        "Rules:\n"
        "1. If user uploads a CSV, Excel, or JSON file and asks any question -> trigger DATA or DATA+QA mode.\n"
        "2. If user uploads a PDF with tables and asks about numbers -> trigger DATA mode.\n"
        "3. If both data analysis and drafting a paper are needed -> trigger DATA+PAPER mode.\n\n"
        f"Context:\n{files_context}\n\n"
        "Respond ONLY with one of the following JSON formats: \n"
        "{\"intent\": \"QA\"} or {\"intent\": \"PAPER\"} or {\"intent\": \"BOTH\"} or {\"intent\": \"DATA\"} or {\"intent\": \"DATA+QA\"} or {\"intent\": \"DATA+PAPER\"}."
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        intent = result.get("intent", "QA").upper()
    except Exception as e:
        intent = "QA"  # Fallback
        state["trace_logs"].append({
            "agent": "Orchestrator",
            "status": "warning",
            "message": f"Intent classification failed: {str(e)}. Defaulting to QA."
        })
        
    state["intent"] = intent
    
    # Set dispatch flags
    state["data_analysis_agent"] = "DATA" in intent
    
    state["trace_logs"].append({
        "agent": "Orchestrator",
        "status": "completed",
        "message": f"Classification complete. Intent: {intent}.",
        "data": {
            "intent": intent,
            "profile": profile,
            "data_analysis_agent": state["data_analysis_agent"]
        }
    })
    
    return state
