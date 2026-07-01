from app.agents.orchestrator import orchestrate_session
from app.agents.rag_agent import run_rag_agent
from app.agents.web_research import run_web_research_agent
from app.agents.synthesis import run_synthesis_agent
from app.agents.paper_writer import run_paper_writer_agent
from app.agents.data_analysis_agent import run_data_analysis_agent
from app.agents.humanizer import run_humanizer_agent
from app.agents.memory_engine import run_memory_engine

def run_agent_pipeline(prompt: str, user_id: str, openai_key: str = None, tavily_key: str = None, llm_base_url: str = None, llm_model: str = None, uploaded_data_sheets: list = None) -> dict:
    """
    Executes the multi-agent chain sequentially.
    Returns the final state containing final_output and trace_logs.
    """
    state = {
        "prompt": prompt,
        "user_id": user_id,
        "openai_key": openai_key,
        "tavily_key": tavily_key,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "uploaded_data_sheets": uploaded_data_sheets or [],
        "intent": "QA",
        "data_analysis_agent": False,
        "data_analysis_output": "",
        "memory_profile": {},
        "rag_summary": "",
        "rag_chunks": [],
        "web_papers": [],
        "synthesis_output": "",
        "paper_writer_output": "",
        "final_output": "",
        "trace_logs": []
    }
    
    try:
        # 0. Orchestration (intent & profile)
        state = orchestrate_session(state)
        
        # 1. RAG search
        state = run_rag_agent(state)
        
        # 2. Web Research
        state = run_web_research_agent(state)
        
        # 2.5. Data Analysis Agent (if triggered)
        if state.get("data_analysis_agent"):
            state = run_data_analysis_agent(state)
            
        # 3. Mode branches
        intent = state["intent"]
        if intent == "DATA":
            # Pass data analysis directly to synthesis output for humanizing
            state["synthesis_output"] = state.get("data_analysis_output", "")
        elif intent == "QA" or intent == "DATA+QA":
            state = run_synthesis_agent(state)
        elif intent == "PAPER" or intent == "DATA+PAPER":
            state = run_paper_writer_agent(state)
        elif intent == "BOTH":
            state = run_synthesis_agent(state)
            state = run_paper_writer_agent(state)
            
        # 4. Humanizer
        state = run_humanizer_agent(state)
        
        # 5. Memory Engine (silent)
        state = run_memory_engine(state)
        
    except Exception as e:
        state["trace_logs"].append({
            "agent": "Pipeline Orchestrator",
            "status": "error",
            "message": f"Critical pipeline failure: {str(e)}"
        })
        if not state["final_output"]:
            state["final_output"] = f"An error occurred while executing the research agents: {str(e)}"
            
    return state
