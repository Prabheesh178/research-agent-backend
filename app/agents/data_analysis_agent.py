import os
import json
import numpy as np
from app.database import get_db_connection
from app.utils.data_parser import parse_uploaded_data
from app.utils.vector_store import get_openai_client
from app.config import settings

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")

def run_data_analysis_agent(state: dict) -> dict:
    """
    Data Analysis Agent:
    - Parses CSV, Excel, JSON, TSV files
    - Profiles dataset and checks data quality (duplicates, nulls, outliers)
    - Performs statistical and trend analysis via LLM grounding
    - Recommends best visualizations
    """
    state["trace_logs"].append({
        "agent": "Data Analysis Agent",
        "status": "running",
        "message": "Parsing data sheets and performing quality checks..."
    })
    
    user_id = state["user_id"]
    prompt = state["prompt"]
    openai_key = state["openai_key"]
    memory_profile = state["memory_profile"]
    
    # 1. Retrieve list of files for this user
    files_to_analyze = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM documents WHERE user_id = ?", (user_id,))
        files_to_analyze = [
            r["filename"] for r in cursor.fetchall()
            if r["filename"].lower().endswith((".csv", ".tsv", ".json", ".xlsx", ".xls"))
        ]
        conn.close()
    except Exception as e:
        print(f"Error querying user documents: {str(e)}")
        
    if not files_to_analyze:
        # Fallback: check if PDF is uploaded and user asks for data
        state["data_analysis_output"] = "No tabular data files (.csv, .xlsx, .json, .tsv) found in your workspace to analyze."
        state["trace_logs"].append({
            "agent": "Data Analysis Agent",
            "status": "completed",
            "message": "No data files found to process. Data Analysis Agent idle.",
            "data": {"summary": "Idle"}
        })
        return state
        
    # We will analyze the first data file found in the workspace
    filename = files_to_analyze[0]
    
    # 2. Retrieve parsed data from in-memory state or local disk fallback
    parsed = None
    for sheet in state.get("uploaded_data_sheets", []):
        if sheet.get("filename") == filename:
            parsed = sheet.get("parsed_data")
            break
            
    if not parsed:
        file_path = os.path.join(UPLOADS_DIR, user_id, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                parsed = parse_uploaded_data(file_bytes, filename)
            except Exception as e:
                print(f"Error parsing local fallback data file {filename}: {str(e)}")

    if not parsed:
        state["final_output"] = "It looks like the session expired and your documents need to be re-uploaded. Please upload your data files again."
        state["trace_logs"].append({
            "agent": "Data Analysis Agent",
            "status": "error",
            "message": f"Data file {filename} could not be resolved from memory or disk."
        })
        raise ValueError(state["final_output"])
        
    # 3. Perform Data Cleaning Check
    data_rows = parsed.get("data", [])
    columns = parsed.get("columns", [])
    rows_count = parsed.get("rows_count", 0)
    
    duplicates = 0
    missing_values = {col["name"]: 0 for col in columns}
    outliers = []
    
    # Calculate duplicates and missing values
    seen_rows = set()
    for row in data_rows:
        row_str = json.dumps(row, sort_keys=True)
        if row_str in seen_rows:
            duplicates += 1
        seen_rows.add(row_str)
        
        for col in columns:
            col_name = col["name"]
            val = str(row.get(col_name, "")).strip().lower()
            if val == "" or val in ("nan", "null", "none", "n/a"):
                missing_values[col_name] += 1
                
    # Detect outliers for numeric columns (values > 3 std dev from mean)
    for col in columns:
        if col["type"] == "numeric" and len(data_rows) > 3:
            col_name = col["name"]
            numeric_vals = []
            for row in data_rows:
                try:
                    val = float(row.get(col_name, ""))
                    numeric_vals.append(val)
                except ValueError:
                    pass
            if numeric_vals:
                mean = np.mean(numeric_vals)
                std = np.std(numeric_vals)
                if std > 0:
                    for idx, val in enumerate(numeric_vals):
                        if abs(val - mean) > 3 * std:
                            outliers.append(f"Row {idx+1} in '{col_name}' has outlier: {val} (mean={mean:.2f}, std={std:.2f})")
                            
    # Build data quality note
    quality_issues = []
    clean_status = True
    
    null_cols = [f"{col}: {count}" for col, count in missing_values.items() if count > 0]
    if null_cols:
        quality_issues.append(f"Missing values found: {', '.join(null_cols)}")
        clean_status = False
    if duplicates > 0:
        quality_issues.append(f"{duplicates} duplicate rows identified.")
        clean_status = False
    if outliers:
        quality_issues.append(f"{len(outliers)} outlier values detected (e.g., {outliers[0]})" if len(outliers) > 1 else outliers[0])
        clean_status = False
        
    data_quality_note = "Data looks clean." if clean_status else "; ".join(quality_issues)
    
    # Internal profile
    dataset_profile = {
        "file_type": parsed["file_type"],
        "rows": rows_count,
        "columns": [f"{c['name']} ({c['type']})" for c in columns],
        "missing_values": {c: count for c, count in missing_values.items() if count > 0},
        "sheets": parsed.get("sheets", [])
    }
    
    state["trace_logs"].append({
        "agent": "Data Analysis Agent",
        "status": "info",
        "message": f"Data profile compiled: {rows_count} rows, {len(columns)} columns.",
        "data": {
            "profile": dataset_profile,
            "quality": data_quality_note
        }
    })
    
    # 4. Perform LLM analysis
    client = get_openai_client(openai_key, state.get("llm_base_url"))
    
    # Format a sample of dataset rows for LLM context (up to 40 rows to avoid blowing token limits)
    sample_rows = data_rows[:40]
    sample_text = json.dumps(sample_rows, indent=2)
    
    system_prompt = (
        "You are ROLE 3C — DATA ANALYSIS AGENT. Your task is to analyze the user's uploaded dataset and write a detailed response.\n\n"
        "Input Details:\n"
        f"- Dataset Profile: {json.dumps(dataset_profile)}\n"
        f"- Data Quality Issues: {data_quality_note}\n"
        f"- File Name: {filename}\n"
        f"- Sample Data (first 40 rows):\n{sample_text}\n\n"
        "Rules:\n"
        "1. Never fabricate numbers. Every figure in the output must come from the actual file.\n"
        "2. If the user asks for something the data doesn't support (e.g. trend but no date column), say so clearly: 'No date/time column found — trend analysis not possible. Available columns are: [list]'.\n"
        "3. Always show specific numbers in findings — never vague statements like 'the values are high'.\n"
        "4. Strict Output Format:\n"
        "   DATA QUALITY NOTE:\n"
        "   [Summary of missing values, duplicates, outliers found — or 'Data looks clean']\n\n"
        "   ANALYSIS:\n"
        "   [Direct answer to what the user asked, with specific numbers, calculations, group comparisons, or averages]\n\n"
        "   KEY INSIGHTS:\n"
        "   • Insight 1 — [specific finding with numbers]\n"
        "   • Insight 2 — [specific finding with numbers]\n"
        "   • Insight 3 — [specific finding with numbers]\n\n"
        "   RECOMMENDED VISUALIZATION:\n"
        "   [Chart type: Line/Bar/Histogram/Scatter/Pie/Heatmap] — [one sentence why it fits this data]\n\n"
        "   DATASET SUMMARY:\n"
        "   • Rows analyzed: [X]\n"
        "   • Columns used: [X]\n"
        "   • File type: [CSV / Excel / JSON / PDF table]"
    )
    
    try:
        response = client.chat.completions.create(
            model=state.get("llm_model") or "gpt-4o",  # Use gpt-4o for math/analysis accuracy
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze Request: {prompt}"}
            ],
            temperature=0.1
        )
        analysis_output = response.choices[0].message.content.strip()
    except Exception as e:
        analysis_output = f"Error performing data analysis: {str(e)}"
        state["trace_logs"].append({
            "agent": "Data Analysis Agent",
            "status": "error",
            "message": f"LLM analysis execution failed: {str(e)}"
        })
        
    state["data_analysis_output"] = analysis_output
    
    state["trace_logs"].append({
        "agent": "Data Analysis Agent",
        "status": "completed",
        "message": "Data analysis, cleaning audit, and visualization recommendations complete."
    })
    
    return state
