# CrossBorder Copilot

CrossBorder Copilot is a conversational support assistant for cross-border ecommerce shipping. It combines a FastAPI/LangGraph chat backend, local DHL document retrieval, SQLite session memory, a notepad tool, retrieval evaluation, and two frontend options.

This is the main project entry point. Module-specific notes live in:

- `backend/src/README.md` for backend API and agent details
- `frontend-react/README.md` for the React UI
- `docs/phase1-rag.md` for the original RAG pipeline milestone

## What To Run

Use the React UI for the full app experience. The static HTML page is kept as a lightweight fallback.

### 1. Install Python Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Configure The Backend

Create a `.env` file in the repository root:

```bash
LLM_BASE_URL=https://your-openai-compatible-endpoint.example/v1
LLM_API_KEY=replace_me
LLM_TEMP=0.1
MAX_TOKENS=10000
DB_PATH=crossborder.db
```

If `LLM_BASE_URL` or `LLM_API_KEY` is missing, the backend still starts and `/health` works, but chat endpoints return a 503 explaining that chat is not configured.

### 3. Start The Backend

From the repository root:

```bash
python backend/src/main.py
```

Backend URL:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

Swagger docs:

```text
http://localhost:8000/docs
```

### 4. Start The React Frontend

In a second terminal:

```bash
cd frontend-react
npm install
npm run dev
```

Open the Vite URL shown in the terminal. The UI defaults to:

```text
http://localhost:8000
```

If the backend is running on another port, update the API Base URL in the left panel.

### 5. Static Frontend Fallback

You can also open this file directly in a browser:

```text
frontend/index.html
```

It still requires the FastAPI backend. Use the connection button before sending a message.

### 6. Starting MCP's

Navigate to the `/mcp` and run  
  `python order.py`
  `python support.py`
  `python checklist.py`
Check the std.out of the terminal to assure servers are running.

## Main Features

- Chat endpoint with generated or reusable `session_id`
- LangGraph agent with OpenAI-compatible LLM configuration
- Local DHL knowledge-base retrieval through ChromaDB
- Notepad tool for session-scoped key/value memory
- Conversation history endpoint
- Retrieval evaluation API and CLI
- React frontend with backend health status and configurable API URL
- Static HTML fallback UI

## Project Layout

```text
.
├── backend/src/              # FastAPI app, LangGraph agent, API routes, SQLite models
├── frontend-react/           # Vite React frontend
├── frontend/index.html       # Static fallback frontend
├── pdf_docs/                 # Local DHL PDF sources
├── html_pages/               # Saved DHL HTML sources
├── chroma_db/                # Local ChromaDB vector index
├── mcp/                      # Optional MCP server experiments/tools
├── parse_local.py            # Parse local PDF/HTML sources
├── ingest.py                 # Chunk/embed/index local documents
├── retriever.py              # Reusable retrieval interface
├── eval_retrieval.py         # Retrieval benchmark
└── docs/phase1-rag.md        # Historical RAG milestone notes
```

## API Summary

```http
GET /health
POST /chat/invoke
GET /chat/history/{session_id}
GET /notepad/{session_id}
POST /evaluation/run
```

Example chat request:

```http
POST /chat/invoke
Content-Type: application/json

{
  "query": "What documents are needed for customs clearance?",
  "session_id": ""
}
```

The response includes a `session_id`. Reuse it for follow-up messages and history lookup.

## RAG Pipeline

The local retrieval layer parses DHL PDFs and saved DHL webpages, cleans text, chunks content with metadata, embeds chunks with SentenceTransformers, and stores them in ChromaDB.

Rebuild the index:

```bash
python ingest.py --reset
```

Run one retrieval query:

```bash
python retriever.py "What documents are needed for customs clearance?" -k 5 --show-text
```

Run retrieval evaluation:

```bash
python eval_retrieval.py -k 5
python eval_retrieval.py -k 10 --verbose
```

## Checks

Backend/Python:

```bash
python -m compileall backend/src mcp parse_local.py ingest.py retriever.py eval_retrieval.py
python eval_retrieval.py -k 1
```

React frontend:

```bash
cd frontend-react
npm run lint
npm run build
```

## Notes For Grading

- Start with this README.
- Use the React frontend unless you specifically want to inspect the static fallback page.
- `/health` is the fastest way to diagnose whether the backend is offline, degraded, or chat-ready.
- The backend can run without LLM credentials only for health checks and non-chat endpoints.
- The RAG index is stored locally in `chroma_db/`; rebuild it with `python ingest.py --reset` if retrieval fails.
