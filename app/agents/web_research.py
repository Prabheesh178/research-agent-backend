import httpx
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from app.utils.vector_store import get_openai_client
from app.config import settings

def run_web_research_agent(state: dict) -> dict:
    """
    Queries arXiv and Semantic Scholar. Fallbacks to Tavily if necessary.
    Records: Title, authors, year, venue, abstract, URL, and relevance reason.
    """
    state["trace_logs"].append({
        "agent": "Web Research Agent",
        "status": "running",
        "message": "Initiating academic web research..."
    })
    
    prompt = state["prompt"]
    openai_key = state["openai_key"]
    base_url = state.get("llm_base_url")
    model = state.get("llm_model") or "gpt-4o-mini"
    tavily_key = state["tavily_key"] or settings.TAVILY_API_KEY
    client = get_openai_client(openai_key, base_url)
    
    # 1. Generate search queries using LLM
    query_prompt = (
        f"Based on this user research query: '{prompt}'\n"
        "Generate two things in JSON format:\n"
        "1. 'arxiv_keywords': a string of 3-5 keywords space-separated for arXiv search.\n"
        "2. 'semantic_scholar_query': a slightly different search phrase (max 8 words) for Semantic Scholar.\n"
        "Output ONLY the raw JSON object."
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a research query planner."},
                      {"role": "user", "content": query_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        queries = json_loads_safe(response.choices[0].message.content)
        arxiv_q = queries.get("arxiv_keywords", prompt)
        s2_q = queries.get("semantic_scholar_query", prompt)
    except Exception:
        arxiv_q = prompt
        s2_q = prompt
        
    state["trace_logs"].append({
        "agent": "Web Research Agent",
        "status": "info",
        "message": f"Generated search params - arXiv: '{arxiv_q}' | Semantic Scholar: '{s2_q}'"
    })
    
    papers = []
    
    # --- TASK 1: arXiv Search ---
    try:
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(arxiv_q)}&start=0&max_results=5"
        arxiv_resp = httpx.get(arxiv_url, timeout=10.0)
        if arxiv_resp.status_code == 200:
            arxiv_papers = parse_arxiv_xml(arxiv_resp.text)
            papers.extend(arxiv_papers)
            state["trace_logs"].append({
                "agent": "Web Research Agent",
                "status": "info",
                "message": f"arXiv search returned {len(arxiv_papers)} papers."
            })
    except Exception as e:
        state["trace_logs"].append({
            "agent": "Web Research Agent",
            "status": "warning",
            "message": f"arXiv search failed: {str(e)}"
        })
        
    # --- TASK 2: Semantic Scholar Search ---
    try:
        s2_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote_plus(s2_q)}&limit=4&fields=title,authors,year,venue,abstract,url"
        headers = {}
        s2_resp = httpx.get(s2_url, headers=headers, timeout=10.0)
        if s2_resp.status_code == 200:
            s2_data = s2_resp.json()
            s2_papers = parse_semantic_scholar_json(s2_data)
            papers.extend(s2_papers)
            state["trace_logs"].append({
                "agent": "Web Research Agent",
                "status": "info",
                "message": f"Semantic Scholar search returned {len(s2_papers)} papers."
            })
    except Exception as e:
        state["trace_logs"].append({
            "agent": "Web Research Agent",
            "status": "warning",
            "message": f"Semantic Scholar search failed: {str(e)}"
        })
        
    # Deduplicate papers based on title similarity or URL
    deduped_papers = deduplicate_papers(papers)
    
    # --- TASK 3: Tavily / DuckDuckGo Fallback Search ---
    # Trigger only if we found fewer than 4 papers from academic sources
    if len(deduped_papers) < 4:
        if tavily_key:
            state["trace_logs"].append({
                "agent": "Web Research Agent",
                "status": "info",
                "message": "Fewer than 4 academic papers found. Triggering Tavily web search fallback..."
            })
            try:
                tavily_url = "https://api.tavily.com/search"
                payload = {
                    "api_key": tavily_key,
                    "query": prompt,
                    "max_results": 4,
                    "search_depth": "advanced"
                }
                t_resp = httpx.post(tavily_url, json=payload, timeout=10.0)
                if t_resp.status_code == 200:
                    t_data = t_resp.json()
                    t_results = parse_tavily_json(t_data)
                    deduped_papers.extend(t_results)
                    deduped_papers = deduplicate_papers(deduped_papers)
                    state["trace_logs"].append({
                        "agent": "Web Research Agent",
                        "status": "info",
                        "message": f"Tavily search returned {len(t_results)} web results."
                    })
            except Exception as e:
                state["trace_logs"].append({
                    "agent": "Web Research Agent",
                    "status": "warning",
                    "message": f"Tavily search failed: {str(e)}. Attempting keyless search..."
                })
                # Attempt DuckDuckGo keyless search if Tavily fails
                ddg_results = free_web_search(prompt)
                deduped_papers.extend(ddg_results)
                deduped_papers = deduplicate_papers(deduped_papers)
        else:
            state["trace_logs"].append({
                "agent": "Web Research Agent",
                "status": "info",
                "message": "Tavily key missing. Triggering DuckDuckGo keyless search fallback..."
            })
            ddg_results = free_web_search(prompt)
            deduped_papers.extend(ddg_results)
            deduped_papers = deduplicate_papers(deduped_papers)

    # 4. Filter out items without URLs, and explain relevance using LLM
    final_papers = []
    valid_papers = [p for p in deduped_papers if p.get("url")]
    
    if valid_papers:
        # Batch relevance classification to make it fast
        relevance_system = (
            "You are a research coordinator. You are given a user request and a list of papers found online. "
            "For each paper, write a 1-sentence relevance explanation: why is this paper useful for addressing the user's prompt? "
            "Respond in a JSON array of strings corresponding to the order of papers given. Do not fabricate. Keep it professional."
        )
        
        papers_summary_text = ""
        for i, p in enumerate(valid_papers[:8]):  # Cap at 8 to save tokens/time
            papers_summary_text += f"[{i}] Title: {p['title']}\nAbstract: {p.get('abstract', '')[:200]}...\n\n"
            
        try:
            rel_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": relevance_system},
                    {"role": "user", "content": f"User Prompt: {prompt}\n\nPapers:\n{papers_summary_text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            rel_data = json_loads_safe(rel_response.choices[0].message.content)
            relevance_reasons = rel_data.get("reasons", [])
            # Fallback if list is returned directly
            if not relevance_reasons and isinstance(rel_data, dict) and "reasons" not in rel_data:
                # sometimes LLM outputs {"0": "reason", ...} or list directly
                relevance_reasons = list(rel_data.values()) if isinstance(rel_data, dict) else rel_data
        except Exception:
            relevance_reasons = []
            
        for idx, p in enumerate(valid_papers[:8]):
            reason = relevance_reasons[idx] if idx < len(relevance_reasons) else "Provides background context on the research topic."
            p["relevance"] = reason
            final_papers.append(p)
            
    state["web_papers"] = final_papers
    
    paper_logs = [{"title": p["title"], "authors": p["authors"], "year": p["year"], "url": p["url"]} for p in final_papers]
    
    state["trace_logs"].append({
        "agent": "Web Research Agent",
        "status": "completed",
        "message": f"Completed research. Found {len(final_papers)} valid papers/web results.",
        "data": {
            "papers": paper_logs
        }
    })
    
    return state

# --- Helper functions ---

def json_loads_safe(text: str):
    import json
    try:
        return json.loads(text)
    except Exception:
        return {}

def parse_arxiv_xml(xml_text: str) -> list:
    papers = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title_node = entry.find("atom:title", ns)
            title = title_node.text.strip().replace("\n", " ") if title_node is not None else "Untitled arXiv Paper"
            
            authors = []
            for author in entry.findall("atom:author", ns):
                name_node = author.find("atom:name", ns)
                if name_node is not None:
                    authors.append(name_node.text.strip())
            authors_str = ", ".join(authors) if authors else "Unknown Authors"
            
            published_node = entry.find("atom:published", ns)
            year = published_node.text[:4] if published_node is not None else "Unknown"
            
            summary_node = entry.find("atom:summary", ns)
            abstract = summary_node.text.strip().replace("\n", " ") if summary_node is not None else ""
            
            id_node = entry.find("atom:id", ns)
            url = id_node.text.strip() if id_node is not None else ""
            
            papers.append({
                "title": title,
                "authors": authors_str,
                "year": year,
                "venue": "arXiv",
                "abstract": abstract,
                "url": url,
                "source": "arxiv"
            })
    except Exception:
        pass
    return papers

def parse_semantic_scholar_json(data: dict) -> list:
    papers = []
    if "data" not in data:
        return papers
    for item in data["data"]:
        title = item.get("title", "Untitled Paper")
        
        authors_list = item.get("authors", [])
        authors = ", ".join([a["name"] for a in authors_list]) if authors_list else "Unknown Authors"
        
        year = str(item.get("year", "")) if item.get("year") else "Unknown"
        venue = item.get("venue", "Semantic Scholar")
        abstract = item.get("abstract", "") or ""
        url = item.get("url", "")
        
        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue,
            "abstract": abstract,
            "url": url,
            "source": "semantic_scholar"
        })
    return papers

def parse_tavily_json(data: dict) -> list:
    results = []
    if "results" not in data:
        return results
    for item in data["results"]:
        title = item.get("title", "Web Result")
        url = item.get("url", "")
        content = item.get("content", "")
        
        results.append({
            "title": title,
            "authors": "Web Article",
            "year": "Recent",
            "venue": "Web Search",
            "abstract": content,
            "url": url,
            "source": "tavily"
        })
    return results

def deduplicate_papers(papers: list) -> list:
    seen_urls = set()
    seen_titles = set()
    unique = []
    
    for p in papers:
        url = p.get("url", "").lower().strip()
        title = p.get("title", "").lower().strip()
        # Clean title spaces
        title = " ".join(title.split())
        
        if url and url in seen_urls:
            continue
        if title in seen_titles:
            continue
            
        if url:
            seen_urls.add(url)
        seen_titles.add(title)
        unique.append(p)
        
    return unique

def free_web_search(query: str) -> list:
    import re
    from urllib.parse import unquote, quote_plus
    import html as html_parser
    
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=10.0)
        if resp.status_code != 200:
            return []
            
        html = resp.text
        results = []
        
        # Simple extraction of titles, links, and snippets
        matches = re.findall(r'<a class="result__a" href="([^"]+)">([^<]+)</a>', html)
        snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', html)
        
        if not snippets:
            snippets = re.findall(r'<div class="result__snippet">([^<]+)</div>', html)
            
        for i, (link, title) in enumerate(matches[:4]):
            snippet = snippets[i] if i < len(snippets) else ""
            if "uddg=" in link:
                link = link.split("uddg=")[1].split("&")[0]
                link = unquote(link)
                
            title = html_parser.unescape(title).strip()
            snippet = html_parser.unescape(snippet).strip()
            
            results.append({
                "title": title,
                "authors": "Web Source",
                "year": "Recent",
                "venue": "Web Search",
                "abstract": snippet,
                "url": link,
                "source": "duckduckgo"
            })
        return results
    except Exception as e:
        print(f"DuckDuckGo search error: {str(e)}")
        return []
