# CrossBorder Copilot

CrossBorder Copilot is a conversational support assistant for cross-border ecommerce shipping.  
This branch extends the existing repository by integrating a presentation-ready React frontend with Ethan’s FastAPI + LangGraph backend, and by exposing the retrieval evaluation workflow through a backend API and frontend dashboard.

---

## What this branch adds

This branch focuses on two major improvements:

1. **Frontend application (React + TypeScript)**
   - Replaces the earlier HTML-only frontend experience with a cleaner React interface
   - Supports live chat with the FastAPI backend
   - Displays session IDs and supports session history loading
   - Shows tool calls used during the conversation
   - Adds an evaluation dashboard directly into the UI

2. **Evaluation integration**
   - Exposes the retrieval benchmark through a FastAPI route
   - Allows evaluation to be triggered from both `/docs` and the frontend
   - Surfaces benchmark results such as total test cases, passed cases, accuracy, confidence levels, and per-query outcomes

---

## How this branch relates to the other branches

### `main`
The `main` branch contains the original RAG-oriented foundation:
- document parsing
- ingestion
- retrieval
- retrieval evaluation
- DHL knowledge base assets

### `ethan`
Ethan’s branch adds the backend application structure:
- FastAPI app
- LangGraph agent
- session-based chat flow
- notepad tool
- optional MCP support
- initial frontend structure

### `nikhil/frontend-eval` (this branch)
This branch builds on Ethan’s backend and connects it with the RAG/evaluation layer from the broader repository.  
It adds:
- a React + TypeScript frontend
- better presentation readiness
- evaluation API support
- evaluation dashboard integration
- fixes needed to make the retriever/evaluation workflow operate correctly from the backend runtime

---

## Architecture

```text
React Frontend
    ↓
FastAPI Backend
    ↓
LangGraph Agent + Tools
    ↓
RAG Retriever / ChromaDB
    ↓
Evaluation Benchmark