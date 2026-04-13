import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { getHistory, sendMessage, type HistoryMessage } from "./api/chat";
import {
  ApiError,
  DEFAULT_API_BASE,
  getHealth,
  getStoredApiBase,
  normalizeApiBase,
  storeApiBase,
  type HealthResponse,
} from "./api/config";
import { runEvaluation } from "./api/eval";

type EvalResult = {
  id: number;
  query: string;
  category: string;
  passed: boolean;
  confidence: string;
  top_score: number;
  source_match?: boolean;
  keyword_match?: boolean;
};

type EvalResponse = {
  total: number;
  passed: number;
  accuracy: number;
  results: EvalResult[];
};

type ConnectionState = "checking" | "ready" | "degraded" | "offline";

const quickPrompts = [
  "What documents are needed for customs clearance?",
  "Can DHL ship lithium batteries internationally?",
  "How do I prepare a commercial invoice?",
  "What peak season or fuel surcharges might apply?",
];

function App() {
  const [apiBaseInput, setApiBaseInput] = useState(() => getStoredApiBase());
  const [apiBase, setApiBase] = useState(() => getStoredApiBase());
  const [connection, setConnection] = useState<ConnectionState>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [notice, setNotice] = useState("");

  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<HistoryMessage[]>([]);
  const [toolCalls, setToolCalls] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);

  const [evalResult, setEvalResult] = useState<EvalResponse | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const chatMessages = useMemo(
    () => messages.filter((message) => message.role !== "system"),
    [messages],
  );

  const checkConnection = useCallback(async (base = apiBase) => {
    const normalized = normalizeApiBase(base);
    setConnection("checking");
    setNotice("");

    try {
      const result = await getHealth(normalized);
      setHealth(result);
      setConnection(result.chat_ready ? "ready" : "degraded");
      if (!result.chat_ready) {
        setNotice(
          result.startup_error ||
            "Backend is reachable, but chat is not ready. Check backend environment variables.",
        );
      }
    } catch (error) {
      setHealth(null);
      setConnection("offline");
      setNotice(formatError(error, `Cannot reach API at ${normalized}.`));
    }
  }, [apiBase]);

  useEffect(() => {
    void checkConnection(apiBase);
  }, [apiBase, checkConnection]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ block: "end" });
  }, [chatMessages.length, loading]);

  function handleApiSubmit(event: FormEvent) {
    event.preventDefault();
    const normalized = storeApiBase(apiBaseInput || DEFAULT_API_BASE);
    setApiBaseInput(normalized);
    setApiBase(normalized);
  }

  function startNewSession() {
    setSessionId("");
    setMessages([]);
    setToolCalls([]);
    setInput("");
    setNotice("");
  }

  async function handleSend(prompt = input) {
    const query = prompt.trim();
    if (!query || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setInput("");
    setLoading(true);
    setNotice("");

    try {
      const response = await sendMessage(apiBase, query, sessionId);
      setSessionId(response.session_id);
      setToolCalls(response.tool_calls_made ?? []);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: response.response || "(empty response)" },
      ]);
      if (connection !== "ready") {
        void checkConnection(apiBase);
      }
    } catch (error) {
      const message = formatError(error, "Error connecting to backend.");
      setNotice(message);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            error instanceof ApiError && error.status === 503
              ? `Backend is running, but chat is disabled: ${message}`
              : message,
        },
      ]);
      void checkConnection(apiBase);
    } finally {
      setLoading(false);
    }
  }

  async function handleLoadHistory() {
    if (!sessionId.trim() || historyLoading) return;

    setHistoryLoading(true);
    setNotice("");
    try {
      const response = await getHistory(apiBase, sessionId);
      setMessages(response.messages);
    } catch (error) {
      setNotice(formatError(error, "Could not load history."));
    } finally {
      setHistoryLoading(false);
    }
  }

  async function handleEval() {
    setEvalLoading(true);
    setNotice("");
    try {
      const response = (await runEvaluation(apiBase)) as EvalResponse;
      setEvalResult(response);
    } catch (error) {
      setNotice(formatError(error, "Evaluation failed."));
    } finally {
      setEvalLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="control-panel" aria-label="Backend and session controls">
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true">
            <Icon name="route" />
          </div>
          <div>
            <h1>CrossBorder Copilot</h1>
            <p>Shipping support workspace</p>
          </div>
        </div>

        <button className="new-chat-button" type="button" onClick={startNewSession}>
          <Icon name="plus" />
          New chat
        </button>

        <form className="api-form" onSubmit={handleApiSubmit}>
          <label htmlFor="apiBase">API base URL</label>
          <div className="api-row">
            <input
              id="apiBase"
              value={apiBaseInput}
              onChange={(event) => setApiBaseInput(event.target.value)}
              placeholder="http://localhost:8000"
              spellCheck={false}
            />
            <button type="submit" aria-label="Use API base URL">
              <Icon name="check" />
            </button>
          </div>
        </form>

        <div className={`status-box ${connection}`}>
          <div className="status-line">
            <span className="status-dot" />
            <strong>{connectionLabel(connection)}</strong>
          </div>
          <p>{healthSummary(connection, health)}</p>
        </div>

        <div className="session-block">
          <span>Session</span>
          <code>{sessionId || "new session"}</code>
          <div className="session-actions">
            <button type="button" onClick={startNewSession}>
              <Icon name="plus" />
              New
            </button>
            <button
              type="button"
              onClick={handleLoadHistory}
              disabled={!sessionId || historyLoading}
            >
              <Icon name="clock" />
              {historyLoading ? "Loading" : "History"}
            </button>
          </div>
        </div>

        <div className="prompt-list">
          <span>Quick prompts</span>
          {quickPrompts.map((prompt) => (
            <button
              key={prompt}
              type="button"
              onClick={() => void handleSend(prompt)}
              disabled={loading}
            >
              <Icon name="message" />
              {prompt}
            </button>
          ))}
        </div>
      </aside>

      <section className="chat-workspace" aria-label="Chat">
        <header className="workspace-header">
          <div>
            <span className="eyebrow">Assistant</span>
            <h2>Ask about shipping, customs, duties, or restricted goods.</h2>
          </div>
          <button className="secondary-button" type="button" onClick={() => void checkConnection(apiBase)}>
            <Icon name="refresh" />
            Check API
          </button>
        </header>

        {notice && <div className="notice">{notice}</div>}

        <div className="message-stream">
          {chatMessages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon" aria-hidden="true">
                <Icon name="spark" />
              </div>
              <strong>Start with a shipment question</strong>
              <p>
                Ask for help with documents, HS codes, duties, restricted goods, or customs steps.
              </p>
              <div className="empty-prompts">
                {quickPrompts.slice(0, 2).map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => void handleSend(prompt)}
                    disabled={loading}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            chatMessages.map((message, index) => (
              <article
                className={`message ${message.role === "user" ? "user" : "assistant"}`}
                key={`${message.role}-${index}`}
              >
                <div className="message-avatar" aria-hidden="true">
                  {message.role === "user" ? <Icon name="user" /> : <Icon name="spark" />}
                </div>
                <div className="message-content">
                  <span>{message.role === "user" ? "You" : "Copilot"}</span>
                  <p>{message.content}</p>
                </div>
              </article>
            ))
          )}
          {loading && (
            <article className="message assistant">
              <div className="message-avatar" aria-hidden="true">
                <Icon name="spark" />
              </div>
              <div className="message-content">
                <span>Copilot</span>
                <p>Thinking...</p>
              </div>
            </article>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void handleSend();
          }}
        >
          <div className="composer-box">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void handleSend();
                }
              }}
              placeholder="Message CrossBorder Copilot"
              rows={1}
            />
            <button type="submit" disabled={loading || !input.trim()} aria-label="Send message">
              <Icon name={loading ? "dots" : "send"} />
            </button>
          </div>
          <p>Enter to send. Shift + Enter for a new line.</p>
        </form>
      </section>

      <aside className="insight-panel" aria-label="Evaluation and tools">
        <section className="panel-section">
          <div className="section-heading">
            <span><Icon name="tool" /> Tools</span>
            <strong>{toolCalls.length ? `${toolCalls.length} used` : "none yet"}</strong>
          </div>
          <div className="tool-list">
            {toolCalls.length ? (
              toolCalls.map((tool) => <code key={tool}>{tool}</code>)
            ) : (
              <p>Tool usage appears here after a response.</p>
            )}
          </div>
        </section>

        <section className="panel-section">
          <div className="section-heading">
            <span><Icon name="chart" /> Retrieval evaluation</span>
            <button type="button" onClick={handleEval} disabled={evalLoading}>
              {evalLoading ? "Running" : "Run"}
            </button>
          </div>

          {!evalResult ? (
            <p className="muted">
              Run the benchmark to verify that the local DHL index is returning useful sources.
            </p>
          ) : (
            <>
              <div className="metric-grid">
                <Metric label="Accuracy" value={`${evalResult.accuracy}%`} />
                <Metric label="Passed" value={`${evalResult.passed}/${evalResult.total}`} />
              </div>
              <div className="eval-list">
                {evalResult.results.map((result) => (
                  <div className="eval-item" key={result.id}>
                    <div>
                      <strong>#{result.id} {result.passed ? "PASS" : "FAIL"}</strong>
                      <span>{result.category}</span>
                    </div>
                    <p>{result.query}</p>
                    <code>{result.confidence} confidence | score {result.top_score}</code>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>
      </aside>
    </main>
  );
}

function Icon({ name }: { name: string }) {
  const common = {
    width: 18,
    height: 18,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  if (name === "plus") return <svg {...common}><path d="M12 5v14" /><path d="M5 12h14" /></svg>;
  if (name === "check") return <svg {...common}><path d="m5 12 4 4L19 6" /></svg>;
  if (name === "clock") return <svg {...common}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></svg>;
  if (name === "message") return <svg {...common}><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z" /></svg>;
  if (name === "refresh") return <svg {...common}><path d="M20 11a8 8 0 0 0-14.8-4" /><path d="M4 5v5h5" /><path d="M4 13a8 8 0 0 0 14.8 4" /><path d="M20 19v-5h-5" /></svg>;
  if (name === "send") return <svg {...common}><path d="m22 2-7 20-4-9-9-4z" /><path d="M22 2 11 13" /></svg>;
  if (name === "dots") return <svg {...common}><path d="M7 12h.01" /><path d="M12 12h.01" /><path d="M17 12h.01" /></svg>;
  if (name === "user") return <svg {...common}><path d="M20 21a8 8 0 0 0-16 0" /><circle cx="12" cy="7" r="4" /></svg>;
  if (name === "tool") return <svg {...common}><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3-3a6 6 0 0 1-8 8l-6 6a2 2 0 0 1-3-3l6-6a6 6 0 0 1 8-8z" /></svg>;
  if (name === "chart") return <svg {...common}><path d="M4 19V5" /><path d="M4 19h16" /><path d="M8 15v-4" /><path d="M12 15V8" /><path d="M16 15v-2" /></svg>;
  if (name === "route") return <svg {...common}><path d="M6 19c0-5 12-3 12-10" /><circle cx="6" cy="19" r="2" /><circle cx="18" cy="5" r="2" /></svg>;
  return <svg {...common}><path d="M12 3l1.6 5.4L19 10l-5.4 1.6L12 17l-1.6-5.4L5 10l5.4-1.6z" /><path d="M19 15l.7 2.3L22 18l-2.3.7L19 21l-.7-2.3L16 18l2.3-.7z" /></svg>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function connectionLabel(connection: ConnectionState): string {
  if (connection === "ready") return "Chat ready";
  if (connection === "degraded") return "Backend degraded";
  if (connection === "offline") return "API offline";
  return "Checking";
}

function healthSummary(connection: ConnectionState, health: HealthResponse | null): string {
  if (connection === "checking") return "Testing the configured backend URL.";
  if (connection === "offline") return "Start FastAPI or update the API base URL.";
  if (!health) return "No health response yet.";
  if (health.chat_ready) return "Backend, database, and chat agent are available.";
  return health.startup_error || "Backend is reachable, but chat is not ready.";
}

function formatError(error: unknown, fallback: string): string {
  if (error instanceof ApiError) return error.detail || fallback;
  if (error instanceof Error) return error.message || fallback;
  return fallback;
}

export default App;
