# React Frontend

Vite React UI for CrossBorder Copilot. The root `README.md` is the main project entry point; this file only documents frontend-specific commands.

## Run

From this directory:

```bash
npm install
npm run dev
```

The UI defaults to:

```text
http://localhost:8000
```

Override it with an environment variable:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

You can also change the API Base URL from the left panel in the UI. This is useful when the backend runs on another port.

## Backend Status

The UI calls `/health` and shows one of three states:

- Chat ready: backend and LLM config are available
- Backend degraded: backend is reachable but chat is not configured
- API offline: frontend cannot reach the configured API URL

## Checks

```bash
npm run lint
npm run build
```

## Related UI

The repository also includes `frontend/index.html`, a static fallback UI that can be opened directly in a browser. Prefer this React UI for demos and grading.
