import os
import sys
from dotenv import load_dotenv

# Ensure backend root is in import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agents import run_agent_pipeline
from app.config import settings

# Load local environment variables
load_dotenv()

def test_agent_routing_and_execution(prompt: str, test_name: str):
    print(f"\n==================================================")
    print(f"TESTING: {test_name}")
    print(f"PROMPT: '{prompt}'")
    print(f"==================================================")
    
    # We will pass the API keys from settings
    openai_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
    tavily_key = os.getenv("TAVILY_API_KEY") or settings.TAVILY_API_KEY
    llm_base_url = os.getenv("LLM_BASE_URL") or settings.LLM_BASE_URL
    llm_model = os.getenv("LLM_MODEL") or settings.LLM_MODEL
    
    if not openai_key:
        print("Error: OPENAI_API_KEY is not configured in settings or environment. Skipping test.")
        return
        
    try:
        state = run_agent_pipeline(
            prompt=prompt,
            user_id="default_academic",
            openai_key=openai_key,
            tavily_key=tavily_key,
            llm_base_url=llm_base_url,
            llm_model=llm_model
        )
        print(f"Classified Intent: {state.get('intent')}")
        print(f"Confidence Scorer Enabled: {state.get('confidence_scorer')}")
        print(f"Export Agent Enabled: {state.get('export_agent')}")
        print("\n--- FINAL OUTPUT ---")
        print(state.get("final_output"))
        print("--------------------")
        
        # Verify trace logs
        print("\nTrace Log Steps:")
        for log in state.get("trace_logs", []):
            print(f"- [{log['agent']}] ({log['status']}): {log['message']}")
            
    except Exception as e:
        print(f"Test failed with exception: {str(e)}")

if __name__ == "__main__":
    # Test 1: Research Gap Finder
    test_agent_routing_and_execution(
        prompt="identify the research gap in combining reinforcement learning with surgical robotics",
        test_name="Gap Finder Intent"
    )
    
    # Test 2: Abstract Grader with Confidence Scoring
    test_agent_routing_and_execution(
        prompt="grade my abstract and score how sure you are: This study introduces Antigravity, a coding assistant. We demonstrate that agentic workflows speed up programming. However, we do not provide empirical stats.",
        test_name="Abstract Grader + Confidence Scorer"
    )
    
    # Test 3: Reference Formatter
    test_agent_routing_and_execution(
        prompt="format my references to IEEE: Smith, J. and Doe, A. 2021. 'A study on agents.' Journal of AI, vol. 12, pp. 34-45.",
        test_name="Reference Formatter Intent"
    )
