import json
from app.utils.vector_store import get_openai_client

def run_citation_graph_agent(state: dict) -> dict:
    state["trace_logs"].append({
        "agent": "Citation Graph Agent",
        "status": "running",
        "message": "Generating citation relationship graph and paper clusters..."
    })
    
    prompt = state["prompt"]
    web_papers = state.get("web_papers", [])
    rag_chunks = state.get("rag_chunks", [])
    client = get_openai_client(state["openai_key"], state.get("llm_base_url"))
    
    # Format current papers
    papers_text = ""
    for idx, paper in enumerate(web_papers):
        papers_text += f"[{idx + 1}] Title: {paper['title']} | Authors: {paper['authors']} | Year: {paper['year']} | Venue: {paper['venue']} | URL: {paper['url']}\n"
    
    system_prompt = (
        "You are an academic Citation Graph Agent. Your job is to:\n"
        "1. Build a relationship map between the retrieved papers.\n"
        "2. Identify: which papers are foundational (highly cited or seminal in this set), which are recent builds on older work, and which share the same methodology or dataset.\n"
        "3. Group papers into thematic clusters (e.g., same topic, same method, same period).\n\n"
        "OUTPUT FORMAT (Strictly match this markdown):\n\n"
        "### FOUNDATIONAL PAPERS\n"
        "- [Paper Title] — [Year] — why it's foundational\n\n"
        "### PAPER CLUSTERS\n"
        "**Cluster 1: [Theme Name]**\n"
        "  - [Paper Title A] → [Paper Title B] (B builds on A)\n"
        "  - [Paper Title A] → [Paper Title C] (C extends A to new domain)\n\n"
        "**Cluster 2: [Theme Name]**\n"
        "  - ...\n\n"
        "### ISOLATED PAPERS\n"
        "- [Paper Title] — [Reason for isolation]\n\n"
        "### VISUALIZATION GRAPH DESIGN\n"
        "```directed_graph\n"
        "Nodes:\n"
        "- Node list...\n"
        "Edges:\n"
        "- Edge connections...\n"
        "```\n\n"
        "Rules:\n"
        "- Only map papers that were actually retrieved. Never invent papers or URLs.\n"
        "- If fewer than 3 papers are retrieved, state: 'Not enough papers for a meaningful graph — run web research first.'"
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User query: {prompt}\n\nRetrieved Papers:\n{papers_text}"}
            ],
            temperature=0.2
        )
        output = response.choices[0].message.content.strip()
    except Exception as e:
        output = f"Failed to build citation graph: {str(e)}"
        
    state["final_output"] = output
    state["trace_logs"].append({
        "agent": "Citation Graph Agent",
        "status": "completed",
        "message": "Citation relationship graph generated."
    })
    return state

def run_gap_finder_agent(state: dict) -> dict:
    state["trace_logs"].append({
        "agent": "Research Gap Finder Agent",
        "status": "running",
        "message": "Analyzing literature for unresolved conflicts and unexplored gaps..."
    })
    
    prompt = state["prompt"]
    web_papers = state.get("web_papers", [])
    rag_summary = state.get("rag_summary", "")
    client = get_openai_client(state["openai_key"], state.get("llm_base_url"))
    
    papers_text = ""
    for idx, paper in enumerate(web_papers):
        papers_text += f"[{idx + 1}] Title: {paper['title']} | Year: {paper['year']} | Abstract: {paper.get('abstract', '')}\n"
        
    system_prompt = (
        "You are an academic PhD Supervisor and Research Gap Finder Agent.\n"
        "Analyze the provided literature context (both web and uploaded PDFs) and systematically identify what has NOT been studied.\n\n"
        "OUTPUT FORMAT (Strictly match this markdown):\n\n"
        "## RESEARCH GAPS IDENTIFIED\n\n"
        "### Gap 1: [Gap Title]\n"
        "- **Description**: [2-3 sentences explaining exactly what has not been done]\n"
        "- **Evidence**: Based on [Paper X] and [Paper Y], neither study explored...\n"
        "- **Opportunity**: A study that [specific proposal] would address this gap.\n"
        "- **Feasibility**: LOW / MEDIUM / HIGH\n\n"
        "### Gap 2: ... (List up to 4 gaps)\n\n"
        "## STRONGEST RECOMMENDATION FOR THESIS\n"
        "[Pick the single most viable gap and explain why it is defensible, original, and feasible.]\n\n"
        "Rules:\n"
        "- Every gap must be supported by what WAS found in the literature — do not make up papers.\n"
        "- Be specific (e.g., 'no study combines GNNs with edge computing for this task' rather than 'more research is needed')."
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Prompt: {prompt}\n\nLocal PDF Summary: {rag_summary}\n\nWeb papers:\n{papers_text}"}
            ],
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
    except Exception as e:
        output = f"Failed to identify research gaps: {str(e)}"
        
    state["final_output"] = output
    state["trace_logs"].append({
        "agent": "Research Gap Finder Agent",
        "status": "completed",
        "message": "Research gaps identified and mapped."
    })
    return state

def run_compare_agent(state: dict) -> dict:
    state["trace_logs"].append({
        "agent": "Multi-Paper Comparison Agent",
        "status": "running",
        "message": "Extracting methodology, datasets, and limitations for structured comparison table..."
    })
    
    prompt = state["prompt"]
    web_papers = state.get("web_papers", [])
    rag_chunks = state.get("rag_chunks", [])
    client = get_openai_client(state["openai_key"], state.get("llm_base_url"))
    
    papers_text = ""
    for idx, paper in enumerate(web_papers):
        papers_text += f"Paper [{idx + 1}]: {paper['title']} ({paper['year']}) | Authors: {paper['authors']} | Abstract: {paper.get('abstract', '')}\n"
    for idx, chunk in enumerate(rag_chunks[:3]):
         papers_text += f"Local Document Chunk [{idx + 4}]: Source: {chunk['filename']}, Page: {chunk['page_num']} | Content: {chunk['text'][:300]}\n"
         
    system_prompt = (
        "You are an academic Comparison Agent. Produce a structured comparison across the provided papers.\n\n"
        "OUTPUT FORMAT (Strictly match this markdown):\n\n"
        "## MULTI-PAPER COMPARISON\n\n"
        "| Dimension | [Paper 1 Title] | [Paper 2 Title] | [Paper 3 Title] |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| **Methodology** | ... | ... | ... |\n"
        "| **Dataset** | ... | ... | ... |\n"
        "| **Approach** | ... | ... | ... |\n"
        "| **Key Result** | ... | ... | ... |\n"
        "| **Limitation** | ... | ... | ... |\n"
        "| **Year** | ... | ... | ... |\n\n"
        "### SYNTHESIS COMPARISON\n"
        "[3-5 sentences summarizing the overall pattern across papers — where they agree, where they diverge, and how the research has evolved.]\n\n"
        "### KEY DIFFERENCES\n"
        "- **Difference 1**: [specific difference with paper titles]\n"
        "- **Difference 2**: [specific difference]\n\n"
        "Rules:\n"
        "- Do not invent parameters. If a dimension is missing in a paper, write 'Not reported'.\n"
        "- Limit the table to 4 papers max for readability."
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Request: {prompt}\n\nAvailable papers:\n{papers_text}"}
            ],
            temperature=0.2
        )
        output = response.choices[0].message.content.strip()
    except Exception as e:
        output = f"Failed to generate comparison table: {str(e)}"
        
    state["final_output"] = output
    state["trace_logs"].append({
        "agent": "Multi-Paper Comparison Agent",
        "status": "completed",
        "message": "Comparison table drafted."
    })
    return state

def run_methodology_agent(state: dict) -> dict:
    state["trace_logs"].append({
        "agent": "Methodology Suggester Agent",
        "status": "running",
        "message": "Evaluating research question to suggest empirical designs and tools..."
    })
    
    prompt = state["prompt"]
    memory_profile = state.get("memory_profile", {})
    client = get_openai_client(state["openai_key"], state.get("llm_base_url"))
    
    system_prompt = (
        "You are an academic Methodology Suggester Agent. Recommend the most appropriate research methodology with justifications and citations.\n\n"
        "OUTPUT FORMAT (Strictly match this markdown):\n\n"
        "## RECOMMENDED METHODOLOGY: [Methodology Name]\n\n"
        "### RATIONALE FOR THIS APPROACH\n"
        "[3-4 sentences linking the suggested methodology to the user's specific research question]\n\n"
        "### METHODOLOGY STRUCTURE\n"
        "- **Design Type**: Quantitative / Qualitative / Mixed Methods\n"
        "- **Design Detail**: Experimental / Survey / Case Study / Systematic Review / etc.\n"
        "- **Data Collection Protocol**: [specific data gathering description]\n"
        "- **Analysis Plan**: [statistical tests or qualitative analysis models]\n"
        "- **Recommended Tools**: [software, frameworks, libraries, platforms]\n\n"
        "### ALTERNATIVE METHODOLOGIES\n"
        "1. **[Alternative 1]** — Use this if [condition/constraint changes]\n"
        "2. **[Alternative 2]** — Use this if [condition/constraint changes]\n\n"
        "### MITIGATING LIMITATIONS\n"
        "- **Limitation 1**: [Description] -> *Mitigation*: [Specific solution]\n"
        "- **Limitation 2**: [Description] -> *Mitigation*: [Specific solution]"
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Prompt: {prompt}\n\nUser Profile Domain: {memory_profile.get('domain', 'academic')}"}
            ],
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
    except Exception as e:
        output = f"Failed to suggest methodology: {str(e)}"
        
    state["final_output"] = output
    state["trace_logs"].append({
        "agent": "Methodology Suggester Agent",
        "status": "completed",
        "message": "Methodology suggested."
    })
    return state

def run_paraphrase_agent(state: dict) -> dict:
    state["trace_logs"].append({
        "agent": "Plagiarism-Safe Paraphraser Agent",
        "status": "running",
        "message": "Rewriting paragraphs to avoid plagiarism while retaining citations..."
    })
    
    prompt = state["prompt"]
    memory_profile = state.get("memory_profile", {})
    client = get_openai_client(state["openai_key"], state.get("llm_base_url"))
    
    system_prompt = (
        "You are an academic Plagiarism-Safe Paraphraser Agent.\n"
        "Produce 3 distinct paraphrased versions of the input text.\n\n"
        "Rules:\n"
        "- Preserve the original meaning completely.\n"
        "- Use completely different sentence structures for each version.\n"
        "- Preserve citation markers (e.g., [1], [PDF: filename]) exactly as they appear in the original text.\n"
        "- Match vocabulary level and connector preferences.\n\n"
        "OUTPUT FORMAT (Strictly match this markdown):\n\n"
        "### ORIGINAL TEXT (For Reference)\n"
        "[Original text]\n\n"
        "### VERSION 1 — FORMAL / DIRECT\n"
        "[Paraphrased version 1]\n\n"
        "### VERSION 2 — ANALYTICAL / EXPANDED\n"
        "[Paraphrased version 2]\n\n"
        "### VERSION 3 — CONCISE / SUMMARY\n"
        "[Paraphrased version 3]\n\n"
        "### SIMILARITY ASSESSMENT\n"
        "- Version 1 structurally distinct: Yes ✓\n"
        "- Version 2 structurally distinct: Yes ✓\n"
        "- Version 3 structurally distinct: Yes ✓"
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to paraphrase: {prompt}\n\nVocabulary Level: {memory_profile.get('vocab_level', 'postgraduate')}"}
            ],
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
    except Exception as e:
        output = f"Failed to paraphrase text: {str(e)}"
        
    state["final_output"] = output
    state["trace_logs"].append({
        "agent": "Plagiarism-Safe Paraphraser Agent",
        "status": "completed",
        "message": "Paraphrased text generated."
    })
    return state

def run_abstract_grader_agent(state: dict) -> dict:
    state["trace_logs"].append({
        "agent": "Abstract Grader Agent",
        "status": "running",
        "message": "Evaluating abstract structure and grading criteria..."
    })
    
    prompt = state["prompt"]
    memory_profile = state.get("memory_profile", {})
    client = get_openai_client(state["openai_key"], state.get("llm_base_url"))
    
    system_prompt = (
        "You are an academic Abstract Grader Agent.\n"
        "Score the abstract against academic standards (out of 80 points total) and give specific actionable feedback.\n\n"
        "SCORING CRITERIA (score each out of 10):\n"
        "1. Problem statement clarity\n"
        "2. Gap identification\n"
        "3. Proposed approach\n"
        "4. Key results specificity\n"
        "5. Implications/Conclusive take-away\n"
        "6. Word count compliance (Target 150-250 words)\n"
        "7. Citation hygiene (Should contain zero citations)\n"
        "8. Self-contained comprehensibility\n\n"
        "OUTPUT FORMAT (Strictly match this markdown):\n\n"
        "## ABSTRACT EVALUATION REPORT\n"
        "**OVERALL SCORE**: [Score]/80\n\n"
        "| Evaluation Criterion | Score | Feedback |\n"
        "| :--- | :--- | :--- |\n"
        "| Problem Statement | X/10 | ... |\n"
        "| Gap Identification | X/10 | ... |\n"
        "| Proposed Approach | X/10 | ... |\n"
        "| Key Results | X/10 | ... |\n"
        "| Implication | X/10 | ... |\n"
        "| Word Count | X/10 | ... |\n"
        "| No Citations | X/10 | ... |\n"
        "| Self-contained | X/10 | ... |\n\n"
        "### TOP 3 CRITICAL IMPROVEMENTS\n"
        "1. **Improvement 1**: [Explanation with example rewrite]\n"
        "2. **Improvement 2**: [Explanation with example]\n"
        "3. **Improvement 3**: [Explanation with example]\n\n"
        "### PROPOSED REWRITTEN VERSION\n"
        "[Fully rewritten academic abstract incorporating all the improvements]"
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Abstract text: {prompt}"}
            ],
            temperature=0.2
        )
        output = response.choices[0].message.content.strip()
    except Exception as e:
        output = f"Failed to grade abstract: {str(e)}"
        
    state["final_output"] = output
    state["trace_logs"].append({
        "agent": "Abstract Grader Agent",
        "status": "completed",
        "message": "Abstract evaluated and scored."
    })
    return state

def run_ref_formatter_agent(state: dict) -> dict:
    state["trace_logs"].append({
        "agent": "Reference Formatter Agent",
        "status": "running",
        "message": "Parsing bibliography and cleaning references..."
    })
    
    prompt = state["prompt"]
    memory_profile = state.get("memory_profile", {})
    client = get_openai_client(state["openai_key"], state.get("llm_base_url"))
    
    target_style = memory_profile.get("citation_style", "IEEE")
    
    system_prompt = (
        f"You are a Reference Formatter Agent. Clean and reformat every citation/reference into exact {target_style} style.\n"
        "Fix capitalization, punctuation, ordering, and missing fields.\n\n"
        "OUTPUT FORMAT (Strictly match this markdown):\n\n"
        f"## CLEANED BIBLIOGRAPHY ({target_style} Style)\n\n"
        "[1] ...\n"
        "[2] ...\n\n"
        "### AUDIT SUMMARY\n"
        "- Number of references cleaned: X\n"
        "- Capitalization fixes: X\n"
        "- Missing DOI/URL flags: X\n\n"
        "### REFERENCES NEEDING MANUAL VERIFICATION\n"
        "- **Reference [X]**: [Explanation of what is missing, e.g., year, publisher, URL]"
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"References list:\n{prompt}"}
            ],
            temperature=0.1
        )
        output = response.choices[0].message.content.strip()
    except Exception as e:
        output = f"Failed to format references: {str(e)}"
        
    state["final_output"] = output
    state["trace_logs"].append({
        "agent": "Reference Formatter Agent",
        "status": "completed",
        "message": "Bibliography reformatted."
    })
    return state

def run_proposal_writer_agent(state: dict) -> dict:
    state["trace_logs"].append({
        "agent": "Research Proposal Writer Agent",
        "status": "running",
        "message": "Drafting research proposal sections and objectives..."
    })
    
    prompt = state["prompt"]
    web_papers = state.get("web_papers", [])
    rag_summary = state.get("rag_summary", "")
    client = get_openai_client(state["openai_key"], state.get("llm_base_url"))
    
    papers_text = ""
    for idx, paper in enumerate(web_papers[:3]):
        papers_text += f"[{idx + 1}] {paper['title']} ({paper['year']}) | Abstract: {paper.get('abstract', '')}\n"
        
    system_prompt = (
        "You are an academic Research Proposal Writer Agent. Write a structured research proposal.\n\n"
        "OUTPUT FORMAT (Strictly match this markdown):\n\n"
        "# RESEARCH PROPOSAL\n\n"
        "## 1. TITLE\n"
        "[Academic, descriptive title]\n\n"
        "## 2. BACKGROUND & CONTEXT\n"
        "[300-400 words outlining the scientific context, referencing cited works]\n\n"
        "## 3. PROBLEM STATEMENT\n"
        "[150-200 words defining the core research challenge and why it needs addressing]\n\n"
        "## 4. RESEARCH QUESTIONS\n"
        "- **RQ1**: [Question]\n"
        "- **RQ2**: [Question]\n\n"
        "## 5. RESEARCH OBJECTIVES\n"
        "- **Objective 1**: [Investigate/Evaluate/Develop...]\n"
        "- **Objective 2**: [Compare/Assess...]\n\n"
        "## 6. SIGNIFICANCE & ORIGINAL CONTRIBUTION\n"
        "[Description of the theoretical and practical value, detailing the gap it fills]\n\n"
        "## 7. METHODOLOGY SUGGESTION\n"
        "[Brief methodological approach summary]\n\n"
        "## 8. TIMELINE\n"
        "| Phase | Activity | Duration |\n"
        "| :--- | :--- | :--- |\n"
        "| Phase 1 | Literature Review & Setup | Month 1-2 |\n"
        "| Phase 2 | Implementation & Experimentation | Month 3-5 |\n"
        "| Phase 3 | Analysis & Writing | Month 6-8 |\n\n"
        "## 9. REFERENCES\n"
        "[IEEE/APA Reference listing of cited context]"
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Topic: {prompt}\n\nPDF context: {rag_summary}\n\nWeb papers:\n{papers_text}"}
            ],
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
    except Exception as e:
        output = f"Failed to generate research proposal: {str(e)}"
        
    state["final_output"] = output
    state["trace_logs"].append({
        "agent": "Research Proposal Writer Agent",
        "status": "completed",
        "message": "Research proposal drafted."
    })
    return state

def run_confidence_scorer(state: dict) -> dict:
    """
    Evaluates final output claims, runs an auto-revision loop if confidence is medium/low,
    and appends a confidence report. Supports human approval bypass commands.
    """
    state["trace_logs"].append({
        "agent": "Confidence Scorer",
        "status": "running",
        "message": "Scanning claims and calculating confidence score..."
    })
    
    content = state.get("final_output", "")
    if not content:
        return state
        
    client = get_openai_client(state["openai_key"], state.get("llm_base_url"))
    
    # Check if user explicitly bypasses the review loop (e.g. prompt contains 'approve' or 'override')
    prompt_lower = state.get("prompt", "").lower().strip()
    approve_override = any(kw in prompt_lower for kw in ["approve", "override", "force", "bypass"])
    
    if approve_override:
        state["trace_logs"].append({
            "agent": "Confidence Scorer",
            "status": "info",
            "message": "User approval override detected. Bypassing self-correction review loops."
        })
        max_loops = 1
    else:
        max_loops = 2
        
    current_loop = 1
    state["needs_human_approval"] = False
    
    while current_loop <= max_loops:
        system_prompt = (
            "You are an academic Confidence Scorer Agent.\n"
            "Analyze the provided draft and score its factual assertions.\n"
            "Compare each claim against what a verified reference supports.\n"
            "Factual claims must be classified as:\n"
            "- [VERIFIED]: Directly backed by a RAG/Web citation.\n"
            "- [PARTIAL]: Partially supported; references exist but don't fully verify.\n"
            "- [INFERRED]: Reasoning only; no direct citations support it.\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            "  \"annotated_draft\": \"[draft with claims annotated inline like: 'The model achieves 95% accuracy [VERIFIED]...']\",\n"
            "  \"verified_count\": X,\n"
            "  \"partial_count\": X,\n"
            "  \"inferred_count\": X,\n"
            "  \"confidence_rating\": \"HIGH\" | \"MEDIUM\" | \"LOW\",\n"
            "  \"low_confidence_claims_summary\": \"[summary of claims that lack strong source backing]\"\n"
            "}"
        )
        
        try:
            response = client.chat.completions.create(
                model=state.get("llm_model") or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Draft content to evaluate:\n\n{content}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content)
            
            annotated_draft = result.get("annotated_draft", "")
            verified_count = result.get("verified_count", 0)
            partial_count = result.get("partial_count", 0)
            inferred_count = result.get("inferred_count", 0)
            confidence_rating = result.get("confidence_rating", "HIGH").upper()
            low_claims = result.get("low_confidence_claims_summary", "")
            
            # Loop back for self-correction rewrite if confidence is LOW/MEDIUM and we have remaining iterations
            if confidence_rating in ["MEDIUM", "LOW"] and current_loop < max_loops:
                state["trace_logs"].append({
                    "agent": "Confidence Scorer",
                    "status": "info",
                    "message": f"Confidence is {confidence_rating} (Inferred: {inferred_count}). Looping back for revision."
                })
                
                revision_prompt = (
                    "You are an Academic Self-Correction Agent. Your task is to rewrite the draft to align it strictly with verified sources and remove or ground any unverified/inferred claims.\n"
                    f"Unverified claims to fix: {low_claims}\n\n"
                    "Ensure every key assertion has a citation or is rephrased to sound like model reasoning rather than a fake factual assertion."
                )
                
                revision_response = client.chat.completions.create(
                    model=state.get("llm_model") or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": revision_prompt},
                        {"role": "user", "content": f"Original Draft:\n\n{content}"}
                    ],
                    temperature=0.2
                )
                content = revision_response.choices[0].message.content.strip()
                current_loop += 1
                continue
                
            # If after loop we still have LOW/MEDIUM confidence, flag for manual review
            if confidence_rating in ["MEDIUM", "LOW"] and not approve_override:
                state["needs_human_approval"] = True
                
            # Create final report
            approval_msg = ""
            if state["needs_human_approval"]:
                approval_msg = (
                    "\n\n> ⚠️ **MANUAL APPROVAL REQUIRED**: The confidence score is still MEDIUM/LOW. "
                    "Review the annotated claims above. To override and accept this draft, reply with **'approve'**."
                )
                
            report = (
                "---\n"
                "### 📊 CONFIDENCE REPORT\n"
                f"- **Verified Claims**: {verified_count}\n"
                f"- **Partially Supported**: {partial_count}\n"
                f"- **Model Inferences**: {inferred_count}\n"
                f"- **Overall Confidence Rating**: **{confidence_rating} ({'Verified >80%' if confidence_rating == 'HIGH' else '50-80%' if confidence_rating == 'MEDIUM' else '<50%'})**\n\n"
                f"*Confidence review loop completed (iterations: {current_loop}).*"
                f"{approval_msg}"
            )
            state["final_output"] = annotated_draft + "\n\n" + report
            break
            
        except Exception as e:
            print(f"Confidence scorer loop iteration {current_loop} failed: {str(e)}")
            state["final_output"] = content + "\n\n---\n### 📊 CONFIDENCE REPORT\n- Review failed: fallback applied."
            break
            
    state["trace_logs"].append({
        "agent": "Confidence Scorer",
        "status": "completed",
        "message": f"Confidence report and loop review complete. Human approval required: {state['needs_human_approval']}."
    })
    return state

def run_export_agent(state: dict) -> dict:
    """
    Sanitizes content for raw file exports.
    """
    state["trace_logs"].append({
        "agent": "Export Agent",
        "status": "running",
        "message": "Formatting draft content for file export..."
    })
    
    content = state.get("final_output", "")
    if not content:
        return state
        
    # Strip internal logging details/labels
    lines = content.split("\n")
    cleaned_lines = []
    for line in lines:
        if any(label in line for label in ["DATASET_PROFILE:", "DISPATCH:"]):
            continue
        cleaned_lines.append(line)
        
    state["final_output"] = "\n".join(cleaned_lines)
    state["trace_logs"].append({
        "agent": "Export Agent",
        "status": "completed",
        "message": "Draft content formatted for export."
    })
    return state
