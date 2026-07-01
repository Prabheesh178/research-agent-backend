# Multi-Agent Academic Research Lab — Backend API

This is the FastAPI-based multi-agent backend for the Academic Research Laboratory. It coordinates a pipeline of intelligent agents (RAG, Web Research, Synthesis, 10 Specialized Agents, Confidence Scorer, Humanizer, and Memory Engine) to ingest, research, analyze, and draft academic papers.

---

## 🚀 Key Features

* **Multi-Agent Pipeline Architecture**: Runs sequential execution blocks (Orchestration → RAG → Web Research → Data Analysis → Synthesis/Specialist → Confidence Scorer → Humanizer → Memory Engine DNA merger).
* **Self-Correcting Confidence Loop**: Inspects factual assertions. If low-confidence claims are identified, it automatically runs an LLM revision step to align text with verified sources, allowing human bypass overrides (reply with `"approve"`).
* **Lightweight Cosine Similarity Database**: SQLite/Python-based vector search storing embeddings for local PDFs without binary C++ compilation dependencies.
* **Academic Web Scraping**: Free-tier APIs (arXiv XML API, Semantic Scholar) with Tavily/DuckDuckGo keyless fallbacks.
* **Profile DNA Memory Engine**: Adapts writing metrics (vocab level, connector preference, sentence variety) over multiple sessions.
* **Security & In-Memory Operations**: Blocked API key leakage query triggers and in-memory tabular data parser (no local disk writes).

---

## 🛠️ Local Setup & Run

### 1. Prerequisites
Make sure you have **Python 3.8+** installed. We recommend using a virtual environment (Conda or venv).

### 2. Install Dependencies
Run from the `backend/` directory:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the `backend/` directory:
```env
# Vector store config
VECTOR_STORE=sqlite

# SQLite Database for profile DNA memory and PDF vectors
DATABASE_URL=sqlite:///./memory.db

# Default Local Server Port
PORT=8000
```
*Note: OpenAI and Tavily API keys are passed dynamically by the client at runtime from the settings panel for maximum isolation.*

### 4. Running the Server
Start the local server with hot-reload:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 📂 Codebase Structure

```text
backend/
├── app/
│   ├── main.py                 # FastAPI gateways & routes (CORS, key routing)
│   ├── config.py               # Settings loader
│   ├── database.py             # SQLite profile/PDF chunks schema
│   ├── utils/
│   │   ├── pdf_parser.py       # pypdf text chunk splitter
│   │   └── vector_store.py     # Cosine similarity calculations
│   └── agents/
│       ├── __init__.py         # Pipeline coordinator
│       ├── orchestrator.py     # Intent classification & profile DNA loading
│       ├── rag_agent.py        # SQLite vector context matching
│       ├── web_research.py     # arXiv/Semantic Scholar/Tavily search
│       ├── specialized_agents.py # 10 specialized agent roles & loops
│       ├── humanizer.py        # Tone & burstiness adaptors
│       └── memory_engine.py    # Merges 30% session DNA with 70% saved DNA
├── requirements.txt
└── README.md
```

---

## 🔗 Main API Routes

* **`GET /ping`**: Returns awake status `{"status": "awake"}` for cold-starts.
* **`POST /api/query`**: Executes the agent pipeline. Accepts parameters:
  * `prompt`: The user's query or draft instruction.
  * `user_id`: Unique identifier (e.g., `default_academic`).
  * `openai_key` / `tavily_key`: Runtime keys.
  * `llm_base_url` / `llm_model`: Custom providers (e.g., OpenRouter).
  * `files[]`: Multi-file uploads (PDF/CSV/Excel/JSON).
* **`GET /api/profile/{user_id}`**: Retrieves the current writing DNA profile.
* **`PUT /api/profile/{user_id}`**: Updates the writing DNA profile manually.
* **`GET /api/documents/{user_id}`**: Lists all uploaded context PDFs.
* **`DELETE /api/documents/{user_id}/{doc_id}`**: Deletes a context PDF.
