from app.database import (
    get_user_profile,
    save_user_profile,
    DEFAULT_PROFILE,
    get_db_connection,
    get_user_skills,
    get_user_plugins
)
from app.utils.vector_store import get_openai_client

def orchestrate_session(state: dict) -> dict:
    """
    Step 1: Classify intent (QA, PAPER, BOTH)
    Step 2: Load user memory profile
    """
    # Auto-sanitize LLM configurations in state for OpenRouter keys
    openai_key = state.get("openai_key") or ""
    llm_base_url = state.get("llm_base_url") or ""
    llm_model = state.get("llm_model") or ""
    
    if openai_key.strip().startswith("sk-or-v1"):
        if not llm_base_url or llm_base_url.strip() == "":
            state["llm_base_url"] = "https://openrouter.ai/api/v1"
        if not llm_model or llm_model.strip() in ["", "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]:
            state["llm_model"] = "google/gemma-2-9b-it:free"

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
    
    # Load Plugins Loader (register active plugins)
    active_plugins = get_user_plugins(user_id)
    state["active_plugins"] = [p["plugin_id"] for p in active_plugins if p.get("enabled", True)]
    state["active_plugin_tools"] = []
    for p in active_plugins:
        if p.get("enabled", True):
            state["active_plugin_tools"].extend(p["tools_provided"])
            
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
        
    prompt = state["prompt"]
    prompt_lower = prompt.lower().strip()
    
    # Skill Router check
    active_skills = get_user_skills(user_id)
    matched_skill = None
    for skill in active_skills:
        if skill.get("enabled", True):
            for kw in skill["trigger_keywords"]:
                if kw.lower() in prompt_lower:
                    matched_skill = skill
                    break
            if matched_skill:
                break
                
    if matched_skill:
        intent = matched_skill["intent_type"]
        state["intent"] = intent
        state["system_prompt_extension"] = matched_skill["system_prompt_extension"]
        state["trace_logs"].append({
            "agent": "Orchestrator",
            "status": "info",
            "message": f"Skill trigger detected! Active skill: '{matched_skill['name']}' (intent: {intent})."
        })
        
        # Disable all default pipeline agents
        state["data_analysis_agent"] = False
        state["citation_graph_agent"] = False
        state["gap_finder_agent"] = False
        state["compare_agent"] = False
        state["methodology_agent"] = False
        state["paraphrase_agent"] = False
        state["abstract_grader_agent"] = False
        state["ref_formatter_agent"] = False
        state["proposal_writer_agent"] = False
        state["export_agent"] = any(kw in prompt_lower for kw in ["export to", "download as", "save as", "export this"])
        state["confidence_scorer"] = any(kw in prompt_lower for kw in ["how sure", "is this accurate", "verified", "confidence report", "how confident"])
        return state
        
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
        "You are an Orchestrator Agent. Your task is to classify the user's research query into one of these intents:\n"
        "- 'QA': User wants an answer, summary, explanation, or search for citations. Keywords: 'what', 'explain', 'summarize', 'find papers', 'compare', 'how does'.\n"
        "- 'PAPER': User wants an academic paper, section, abstract, lit review, or draft written. Keywords: 'write', 'draft', 'generate', 'create a section', 'IEEE format'.\n"
        "- 'BOTH': User wants both QA research AND a draft paper written.\n"
        "- 'DATA': User wants numerical analysis, charts, statistics, formatting, duplicate cleaning, columns comparison, or insights from data sheets. Keywords: 'analyze', 'analyse', 'analysis', 'chart', 'graph', 'plot', 'visualize', 'trend', 'pattern', 'correlation', 'distribution', 'average', 'mean', 'median', 'sum', 'count', 'max', 'min', 'duplicates', 'missing values'.\n"
        "- 'DATA+QA': User has uploaded data files (CSV, Excel, JSON) and wants an analysis QA response.\n"
        "- 'DATA+PAPER': User wants to analyze data first, and then write an academic paper section based on it.\n"
        "- 'CITE_GRAPH': User wants a citation graph, connection map, or paper mapping. Keywords: 'citation graph', 'how are these papers connected', 'paper map', 'visualize citations'.\n"
        "- 'GAP': User wants research gap finding. Keywords: 'research gap', 'what hasn't been studied', 'find the gap', 'unexplored area'.\n"
        "- 'COMPARE': User wants to compare multiple papers. Keywords: 'compare these papers', 'differences between', 'compare methodologies', 'contrast'.\n"
        "- 'METHODOLOGY': User wants methodology suggestions. Keywords: 'what methodology should I use', 'suggest a method', 'best research approach for'.\n"
        "- 'PARAPHRASE': User wants to paraphrase text. Keywords: 'paraphrase this', 'rewrite this paragraph', 'rephrase without plagiarism', '3 versions of'.\n"
        "- 'GRADE': User wants abstract evaluation. Keywords: 'grade my abstract', 'score this abstract', 'check my abstract', 'improve my abstract'.\n"
        "- 'FORMAT_REFS': User wants reference cleaning/formatting. Keywords: 'format my references', 'clean my bibliography', 'convert to IEEE/APA/MLA', 'fix my citations'.\n"
        "- 'PROPOSAL': User wants a research proposal written. Keywords: 'write a proposal', 'research proposal', 'draft my proposal', 'problem statement', 'research objectives'.\n\n"
        "Rules:\n"
        "1. If user uploads a CSV, Excel, or JSON file and asks any question -> trigger DATA or DATA+QA mode.\n"
        "2. If user uploads a PDF with tables and asks about numbers -> trigger DATA mode.\n"
        "3. Choose the specific intent (GAP, COMPARE, CITE_GRAPH, METHODOLOGY, PARAPHRASE, GRADE, FORMAT_REFS, PROPOSAL) if the prompt focuses primarily on that action.\n\n"
        f"Context:\n{files_context}\n\n"
        "Respond ONLY with a JSON object in this format:\n"
        "{\"intent\": \"INTENT_NAME\"}"
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
    
    # Set dispatch flags for all agents
    state["data_analysis_agent"] = "DATA" in intent
    state["citation_graph_agent"] = intent == "CITE_GRAPH"
    state["gap_finder_agent"] = intent == "GAP"
    state["compare_agent"] = intent == "COMPARE"
    state["methodology_agent"] = intent == "METHODOLOGY"
    state["paraphrase_agent"] = intent == "PARAPHRASE"
    state["abstract_grader_agent"] = intent == "GRADE"
    state["ref_formatter_agent"] = intent == "FORMAT_REFS"
    state["proposal_writer_agent"] = intent == "PROPOSAL"
    
    # Check for EXPORT or CONFIDENCE queries explicitly in the prompt
    state["export_agent"] = (intent == "EXPORT") or any(kw in prompt_lower for kw in ["export to", "download as", "save as", "export this"])
    state["confidence_scorer"] = (intent == "CONFIDENCE") or any(kw in prompt_lower for kw in ["how sure", "is this accurate", "verified", "confidence report", "how confident"])
    
    state["trace_logs"].append({
        "agent": "Orchestrator",
        "status": "completed",
        "message": f"Classification complete. Intent: {intent}.",
        "data": {
            "intent": intent,
            "profile": profile,
            "data_analysis_agent": state["data_analysis_agent"],
            "gap_finder_agent": state["gap_finder_agent"],
            "compare_agent": state["compare_agent"]
        }
    })
    
    return state
