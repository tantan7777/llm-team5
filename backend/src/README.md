# Backend

FastAPI backend for CrossBorder Copilot. The root `README.md` is the main project entry point; this file only documents backend-specific behavior.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python backend/src/main.py
```

Alternative from this directory:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Environment

| Variable          | Default          | Description                         |
|-------------------|------------------|-------------------------------------|
| `LLM_BASE_URL`    | required for chat | OpenAI-compatible LLM endpoint      |
| `LLM_API_KEY`     | required for chat | API key for the LLM                 |
| `LLM_TEMP`        | `0.1`            | LLM temperature                     |
| `MAX_TOKENS`      | `10000`          | Max tokens per LLM response         |
| `DB_PATH`         | `crossborder.db` | SQLite database file path           |
| `MCP_SERVER_URLS` | empty            | Comma-separated MCP SSE server URLs |

If LLM credentials are missing, the app starts in degraded mode. `/health` returns `chat_ready: false`, and `/chat/invoke` returns a 503 with setup details.

## Routes

```http
GET /health
GET /
POST /chat/invoke
GET /chat/history/{session_id}
GET /notepad/{session_id}
POST /evaluation/run
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Chat Contract

Start a new session:

```http
POST /chat/invoke
Content-Type: application/json

{
  "query": "What documents are needed for customs clearance?",
  "session_id": ""
}
```

Continue a session:

```http
POST /chat/invoke
Content-Type: application/json

{
  "query": "What about lithium batteries?",
  "session_id": "<existing-session-id>"
}
```

The first stored history message is a system message:

```text
SESSION_ID: <session-id>
```

The agent uses this value when calling the notepad tool.

## Backend Components

```text
main.py                 # App entry point
app/factory.py          # FastAPI factory, lifespan, health routes
app/core/config.py      # Environment config and system prompt
app/core/agent.py       # LangGraph agent and tool routing
app/api/chat.py         # Chat and history routes
app/api/notepad.py      # Notepad inspection route
app/api/evaluation.py   # Retrieval evaluation route
app/db/database.py      # SQLModel schema and SQLite engine
app/tools/notepad.py    # Session-scoped notepad tool
app/tools/knowledge_base.py # Local DHL retrieval tool
```

## Checks

From the repository root:

```bash
python -m compileall backend/src
```

Smoke-test health behavior without LLM credentials:

```bash
DB_PATH=/tmp/cbc-smoke.db python -c "import sys; sys.path.insert(0, 'backend/src'); from fastapi.testclient import TestClient; from app.factory import create_app; client=TestClient(create_app()); print(client.get('/health').json())"
```
