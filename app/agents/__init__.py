from app.agents.orchestrator import orchestrate_session
from app.agents.rag_agent import run_rag_agent
from app.agents.web_research import run_web_research_agent
from app.agents.synthesis import run_synthesis_agent
from app.agents.paper_writer import run_paper_writer_agent
from app.agents.data_analysis_agent import run_data_analysis_agent
from app.agents.humanizer import run_humanizer_agent
from app.agents.memory_engine import run_memory_engine
from app.agents.specialized_agents import (
    run_citation_graph_agent,
    run_gap_finder_agent,
    run_compare_agent,
    run_methodology_agent,
    run_paraphrase_agent,
    run_abstract_grader_agent,
    run_ref_formatter_agent,
    run_proposal_writer_agent,
    run_confidence_scorer,
    run_export_agent
)

def run_agent_pipeline(
    prompt: str,
    user_id: str,
    openai_key: str = None,
    tavily_key: str = None,
    llm_base_url: str = None,
    llm_model: str = None,
    model_tier: str = "FREE",
    uploaded_data_sheets: list = None
) -> dict:
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
        "model_tier": model_tier,
        "uploaded_data_sheets": uploaded_data_sheets or [],
        "intent": "QA",
        "data_analysis_agent": False,
        "citation_graph_agent": False,
        "gap_finder_agent": False,
        "compare_agent": False,
        "methodology_agent": False,
        "paraphrase_agent": False,
        "abstract_grader_agent": False,
        "ref_formatter_agent": False,
        "proposal_writer_agent": False,
        "export_agent": False,
        "confidence_scorer": False,
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
        
        # 0.5. Free Tier Auto Optimization Planner
        tier = (state.get("model_tier") or "FREE").upper()
        will_run_agents = ["Orchestrator"]
        skipped_agents = []
        state["merge_web_summarizer"] = False
        state["merge_paper_sections"] = False
        
        # Identify planned agents
        if state.get("uploaded_data_sheets"):
            will_run_agents.append("Data Ingestion")
        will_run_agents.append("RAG (PDF Search)")
        
        if state["tavily_key"] or any(kw in state["prompt"].lower() for kw in ["arxiv", "scholar", "web search"]):
            will_run_agents.append("Web Research")
            
        intent = state["intent"]
        if intent in ["QA", "DATA+QA", "BOTH"]:
            will_run_agents.append("Synthesis Agent")
        if intent in ["PAPER", "DATA+PAPER", "BOTH"]:
            will_run_agents.append("Paper Writer")
        elif intent == "CITE_GRAPH" or state.get("citation_graph_agent"):
            will_run_agents.append("Citation Graph Agent")
        elif intent == "GAP" or state.get("gap_finder_agent"):
            will_run_agents.append("Research Gap Finder")
        elif intent == "COMPARE" or state.get("compare_agent"):
            will_run_agents.append("Multi-Paper Comparison")
        elif intent == "METHODOLOGY" or state.get("methodology_agent"):
            will_run_agents.append("Methodology Suggester")
        elif intent == "PARAPHRASE" or state.get("paraphrase_agent"):
            will_run_agents.append("Paraphraser")
        elif intent == "GRADE" or state.get("abstract_grader_agent"):
            will_run_agents.append("Abstract Grader")
        elif intent == "FORMAT_REFS" or state.get("ref_formatter_agent"):
            will_run_agents.append("Reference Formatter")
        elif intent == "PROPOSAL" or state.get("proposal_writer_agent"):
            will_run_agents.append("Proposal Writer")
            
        if state.get("confidence_scorer"):
            will_run_agents.append("Confidence Scorer")
            
        will_run_agents.append("Humanizer")
        will_run_agents.append("Memory Engine")
        
        # Define limits
        tier_limit = 100 # Default paid
        if tier == "FREE":
            tier_limit = 6
        elif tier == "LOCAL":
            tier_limit = 4
            
        # Helper to compute current cost
        def get_current_cost(st):
            cost = 3 # Orchestration (1) + Humanizer (1) + Memory Engine (1)
            if st.get("intent") in ["QA", "DATA+QA", "BOTH"]:
                cost += 1 # Synthesis
            if st.get("intent") in ["PAPER", "DATA+PAPER", "BOTH"]:
                cost += 3 if st.get("merge_paper_sections") else 6
            elif st.get("intent") == "CITE_GRAPH" or st.get("citation_graph_agent"):
                cost += 1
            elif st.get("intent") == "GAP" or st.get("gap_finder_agent"):
                cost += 1
            elif st.get("intent") in ["COMPARE", "METHODOLOGY", "PARAPHRASE", "GRADE", "FORMAT_REFS"]:
                cost += 1
            elif st.get("intent") == "PROPOSAL" or st.get("proposal_writer_agent"):
                cost += 4
                
            if st.get("data_analysis_agent"):
                cost += 1
            if (st.get("tavily_key") or len(st.get("web_papers", [])) > 0) and not st.get("merge_web_summarizer"):
                cost += 1
            if st.get("confidence_scorer"):
                cost += 1
            return cost
            
        # Apply Compressions in priority order
        # Compression 1: Skip Confidence Scorer
        if get_current_cost(state) > tier_limit:
            if state.get("confidence_scorer"):
                state["confidence_scorer"] = False
                if "Confidence Scorer" in will_run_agents:
                    will_run_agents.remove("Confidence Scorer")
                skipped_agents.append("Confidence Scorer")
                
        # Compression 2: Skip Citation Graph
        if get_current_cost(state) > tier_limit:
            if state.get("citation_graph_agent"):
                state["citation_graph_agent"] = False
                if "Citation Graph Agent" in will_run_agents:
                    will_run_agents.remove("Citation Graph Agent")
                skipped_agents.append("Citation Graph")
                
        # Compression 3: Skip Gap Finder
        if get_current_cost(state) > tier_limit:
            if state.get("gap_finder_agent"):
                state["gap_finder_agent"] = False
                if "Research Gap Finder" in will_run_agents:
                    will_run_agents.remove("Research Gap Finder")
                skipped_agents.append("Gap Finder")
                
        # Compression 4: Merge web result summarizer
        if get_current_cost(state) > tier_limit:
            state["merge_web_summarizer"] = True
            if "Web Research" in will_run_agents:
                will_run_agents.remove("Web Research")
                will_run_agents.append("Web Research (Raw results)")
                
        # Compression 5: Merge paper writer sections
        if get_current_cost(state) > tier_limit:
            state["merge_paper_sections"] = True
            if "Paper Writer" in will_run_agents:
                will_run_agents.remove("Paper Writer")
                will_run_agents.append("Paper Writer (3 merged calls)")
                
        # Compression 6: Split across windows (mark simulation flag)
        if get_current_cost(state) > tier_limit:
            state["split_windows"] = True
            will_run_agents.append("Split-window delays (15s)")
            
        # Compile transparency card text
        card_text = ""
        if tier == "FREE":
            card_text = (
                "─────────────────────────────────────\n"
                "⚡ Free Tier — Here's what's running:\n\n"
                f"✅ Will run:   {', '.join(will_run_agents)}\n"
                f"⏭️ Skipped:   {', '.join(skipped_agents) if skipped_agents else 'None'}\n\n"
                "💡 Switch to a paid key for the full pipeline.\n"
                "─────────────────────────────────────\n\n"
            )
        elif tier == "LOCAL":
            card_text = "🖥️ Local Mode — Web search disabled. PDF search only.\n\n"
        else:
            if skipped_agents:
                card_text = f"✅ Pipeline running — some elements customized. Skipped: {', '.join(skipped_agents)}\n\n"
                
        if not skipped_agents and tier in ["FREE", "LOCAL"]:
            card_text = "✅ Full pipeline running — nothing skipped.\n\n"
            
        state["transparency_card"] = card_text
        
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
        elif intent == "CITE_GRAPH":
            state = run_citation_graph_agent(state)
        elif intent == "GAP":
            state = run_gap_finder_agent(state)
        elif intent == "COMPARE":
            state = run_compare_agent(state)
        elif intent == "METHODOLOGY":
            state = run_methodology_agent(state)
        elif intent == "PARAPHRASE":
            state = run_paraphrase_agent(state)
        elif intent == "GRADE":
            state = run_abstract_grader_agent(state)
        elif intent == "FORMAT_REFS":
            state = run_ref_formatter_agent(state)
        elif intent == "PROPOSAL":
            state = run_proposal_writer_agent(state)
            
        # 4. Humanizer
        state = run_humanizer_agent(state)
        
        # 4.5. Confidence Scorer (if triggered)
        if state.get("confidence_scorer"):
            state = run_confidence_scorer(state)
            
        # 4.6. Export Formatter (if triggered)
        if state.get("export_agent"):
            state = run_export_agent(state)
            
        # 5. Memory Engine (silent)
        state = run_memory_engine(state)
        
        # Prepend transparency card to final output
        if state.get("transparency_card") and state.get("final_output"):
            state["final_output"] = state["transparency_card"] + state["final_output"]
            
    except Exception as e:
        state["trace_logs"].append({
            "agent": "Pipeline Orchestrator",
            "status": "error",
            "message": f"Critical pipeline failure: {str(e)}"
        })
        if not state["final_output"]:
            state["final_output"] = f"An error occurred while executing the research agents: {str(e)}"
            
    return state
