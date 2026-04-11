# CrossBorder Copilot

LangGraph-powered cross-border shipping assistant with MCP support and SQLite memory.

## Project Structure

```
crossborder_copilot/
├── main.py                  # Entry point
├── requirements.txt
├── README.md
└── app/
    ├── factory.py           # FastAPI app factory + lifespan
    ├── core/
    │   ├── config.py        # All constants & env-var config
    │   └── agent.py         # LangGraph agent builder
    ├── db/
    │   └── database.py      # SQLModel schema + engine
    ├── tools/
    │   └── notepad.py       # Notepad LangChain tool
    └── api/
        ├── chat.py          # /chat/invoke  and  /chat/history routes
        └── notepad.py       # /notepad route
```

## Setup

```bash
pip install -r requirements.txt
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

| Variable          | Default                                        | Description                          |
|-------------------|------------------------------------------------|--------------------------------------|
| `LLM_BASE_URL`    | `https://rsm-8430-finalproject.bjlkeng.io/v1`  | OpenAI-compatible LLM endpoint       |
| `LLM_API_KEY`     | `1012837405`                                   | API key for the LLM                  |
| `LLM_TEMP`        | `0.1`                                          | LLM temperature                      |
| `MAX_TOKENS`      | `4000`                                         | Max tokens per LLM response          |
| `DB_PATH`         | `crossborder.db`                               | SQLite database file path            |
| `MCP_SERVER_URLS` | *(empty)*                                      | Comma-separated MCP SSE server URLs  |

## API Usage

### Start a new conversation (auto session ID)

```http
POST /chat/invoke
Content-Type: application/json

{ "query": "What HS code covers laptop computers?" }
```

Response includes a generated `session_id` — **save this** for follow-up messages.

### Continue an existing conversation

```http
POST /chat/invoke
Content-Type: application/json

{ "query": "What is the import duty rate into Canada?", "session_id": "<your-session-id>" }
```

### View conversation history

```http
GET /chat/history/<session_id>
```

The **first message** in the history is always a system message in the form:

```
SESSION_ID: <your-session-id>
```

Copy the value directly from there — no need to ask the bot for its session ID.

### View notepad

```http
GET /notepad/<session_id>
```

### Swagger UI

Visit `http://localhost:8000/docs` for the full interactive API documentation.
