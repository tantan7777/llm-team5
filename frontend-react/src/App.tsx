import { useState } from "react";
import { getHistory, sendMessage, type HistoryMessage } from "./api/chat";
import { runEvaluation } from "./api/eval";

type EvalResult = {
  id: number;
  query: string;
  category: string;
  passed: boolean;
  confidence: string;
  top_score: number;
};

type EvalResponse = {
  total: number;
  passed: number;
  accuracy: number;
  results: EvalResult[];
};

function App() {
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<HistoryMessage[]>([]);
  const [toolCalls, setToolCalls] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const [evalResult, setEvalResult] = useState<EvalResponse | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: HistoryMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const res = await sendMessage(input, sessionId);
      setSessionId(res.session_id);
      setToolCalls(res.tool_calls_made ?? []);

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.response },
      ]);

      setInput("");
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error connecting to backend." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadHistory = async () => {
    if (!sessionId.trim()) return;

    try {
      const res = await getHistory(sessionId);
      setMessages(res.messages);
    } catch {
      alert("Could not load history.");
    }
  };

  const handleEval = async () => {
    setEvalLoading(true);
    try {
      const res = await runEvaluation();
      setEvalResult(res);
    } catch {
      alert("Evaluation failed.");
    } finally {
      setEvalLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100%",
        background: "#0a0a0a",
        color: "#f5f5f5",
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif',
        padding: "20px 28px",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: "none",
          margin: "0",
        }}
      >
        <div
          style={{
            marginBottom: 18,
            paddingBottom: 14,
            borderBottom: "1px solid #1f1f1f",
          }}
        >
          <h1
            style={{
              margin: 0,
              fontSize: 40,
              fontWeight: 650,
              letterSpacing: "-0.03em",
              color: "#f8f8f8",
            }}
          >
            CrossBorder Copilot
          </h1>
          <p
            style={{
              marginTop: 8,
              color: "#9a9a9a",
              fontSize: 15,
            }}
          >
            Cross-border shipping assistant with session memory, tool usage, and retrieval evaluation.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1.9fr 1fr",
            gap: 18,
            alignItems: "start",
          }}
        >
          <div
            style={{
              background: "#111111",
              border: "1px solid #202020",
              borderRadius: 24,
              padding: 18,
            }}
          >
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr auto",
                gap: 14,
                alignItems: "center",
                marginBottom: 16,
              }}
            >
              <div
                style={{
                  background: "#0c0c0c",
                  border: "1px solid #212121",
                  borderRadius: 16,
                  padding: "12px 14px",
                }}
              >
                <div
                  style={{
                    fontSize: 12,
                    color: "#8b8b8b",
                    marginBottom: 6,
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                  }}
                >
                  Session
                </div>
                <div
                  style={{
                    fontSize: 14,
                    color: "#f1f1f1",
                    wordBreak: "break-all",
                  }}
                >
                  {sessionId || "New session"}
                </div>
              </div>

              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <button
                  onClick={handleLoadHistory}
                  disabled={!sessionId}
                  style={buttonSecondary}
                >
                  Load History
                </button>
                <button
                  onClick={handleEval}
                  disabled={evalLoading}
                  style={buttonPrimary}
                >
                  {evalLoading ? "Running..." : "Run Evaluation"}
                </button>
              </div>
            </div>

            <div
              style={{
                height: "58vh",
                minHeight: 420,
                overflowY: "auto",
                background: "#0b0b0b",
                border: "1px solid #1d1d1d",
                borderRadius: 20,
                padding: 16,
                marginBottom: 14,
              }}
            >
              {messages.length === 0 ? (
                <div style={{ color: "#888", paddingTop: 8 }}>
                  No messages yet. Start by asking about customs, duties, invoices, HS codes, or shipping restrictions.
                </div>
              ) : (
                messages.map((m, i) => {
                  const isUser = m.role === "user";
                  const isSystem = m.role === "system";

                  return (
                    <div
                      key={i}
                      style={{
                        display: "flex",
                        justifyContent: isUser ? "flex-end" : "flex-start",
                        marginBottom: 12,
                      }}
                    >
                      <div
                        style={{
                          maxWidth: "82%",
                          background: isSystem
                            ? "#181818"
                            : isUser
                            ? "#2a2a2a"
                            : "#151515",
                          border: "1px solid #262626",
                          borderRadius: 18,
                          padding: "12px 14px",
                          lineHeight: 1.55,
                          color: "#f2f2f2",
                          whiteSpace: "pre-wrap",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 11,
                            color: "#9a9a9a",
                            marginBottom: 6,
                            textTransform: "capitalize",
                            letterSpacing: "0.04em",
                          }}
                        >
                          {m.role}
                        </div>
                        <div>{m.content}</div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            <div
              style={{
                background: "#0c0c0c",
                border: "1px solid #212121",
                borderRadius: 16,
                padding: 12,
                marginBottom: 14,
                color: "#b5b5b5",
              }}
            >
              <b style={{ color: "#f2f2f2" }}>Tool calls:</b>{" "}
              {toolCalls.length ? toolCalls.join(", ") : "None"}
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr auto",
                gap: 10,
              }}
            >
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about customs, duties, invoices, HS codes, shipping restrictions..."
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                style={{
                  width: "100%",
                  padding: "14px 16px",
                  borderRadius: 16,
                  border: "1px solid #262626",
                  background: "#0c0c0c",
                  color: "#f5f5f5",
                  outline: "none",
                  fontSize: 14,
                  boxSizing: "border-box",
                }}
              />
              <button
                onClick={handleSend}
                disabled={loading}
                style={buttonPrimary}
              >
                {loading ? "..." : "Send"}
              </button>
            </div>
          </div>

          <div
            style={{
              background: "#111111",
              border: "1px solid #202020",
              borderRadius: 24,
              padding: 18,
              minHeight: "calc(58vh + 120px)",
            }}
          >
            <h3
              style={{
                marginTop: 0,
                marginBottom: 16,
                fontSize: 22,
                fontWeight: 600,
                letterSpacing: "-0.02em",
              }}
            >
              Evaluation Dashboard
            </h3>

            {!evalResult ? (
              <div style={{ color: "#8f8f8f" }}>
                No evaluation run yet. Click <b style={{ color: "#f2f2f2" }}>Run Evaluation</b> to benchmark retrieval performance.
              </div>
            ) : (
              <>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 12,
                    marginBottom: 16,
                  }}
                >
                  <MetricCard label="Accuracy" value={`${evalResult.accuracy}%`} />
                  <MetricCard label="Passed" value={`${evalResult.passed} / ${evalResult.total}`} />
                </div>

                <div style={{ marginBottom: 16 }}>
                  <div
                    style={{
                      height: 10,
                      background: "#0b0b0b",
                      borderRadius: 999,
                      overflow: "hidden",
                      border: "1px solid #222",
                    }}
                  >
                    <div
                      style={{
                        width: `${evalResult.accuracy}%`,
                        height: "100%",
                        background: "linear-gradient(90deg, #4a4a4a, #d6d6d6)",
                      }}
                    />
                  </div>
                </div>

                <div
                  style={{
                    maxHeight: "62vh",
                    overflowY: "auto",
                    borderTop: "1px solid #1f1f1f",
                    paddingTop: 6,
                  }}
                >
                  {evalResult.results.map((r) => (
                    <div
                      key={r.id}
                      style={{
                        padding: "12px 0",
                        borderBottom: "1px solid #1b1b1b",
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          gap: 10,
                          marginBottom: 6,
                          alignItems: "center",
                        }}
                      >
                        <div style={{ fontWeight: 600 }}>
                          #{r.id} {r.passed ? "PASS" : "FAIL"}
                        </div>
                        <div
                          style={{
                            fontSize: 11,
                            padding: "4px 8px",
                            borderRadius: 999,
                            background: r.passed ? "#1f1f1f" : "#2b1616",
                            color: r.passed ? "#d9d9d9" : "#f2b8b8",
                            border: "1px solid #2a2a2a",
                            textTransform: "uppercase",
                            letterSpacing: "0.04em",
                          }}
                        >
                          {r.confidence}
                        </div>
                      </div>

                      <div
                        style={{
                          fontSize: 14,
                          color: "#ebebeb",
                          marginBottom: 6,
                          lineHeight: 1.45,
                        }}
                      >
                        {r.query}
                      </div>

                      <div style={{ fontSize: 12, color: "#8f8f8f" }}>
                        {r.category} · score {r.top_score}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        background: "#0c0c0c",
        border: "1px solid #202020",
        borderRadius: 18,
        padding: 14,
      }}
    >
      <div
        style={{
          fontSize: 12,
          color: "#8b8b8b",
          marginBottom: 6,
          textTransform: "uppercase",
          letterSpacing: "0.06em",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 26,
          fontWeight: 650,
          letterSpacing: "-0.03em",
        }}
      >
        {value}
      </div>
    </div>
  );
}

const buttonPrimary: React.CSSProperties = {
  padding: "12px 16px",
  borderRadius: 14,
  border: "1px solid #3a3a3a",
  background: "#2b2b2b",
  color: "#f4f4f4",
  cursor: "pointer",
  fontWeight: 600,
};

const buttonSecondary: React.CSSProperties = {
  padding: "12px 16px",
  borderRadius: 14,
  border: "1px solid #2b2b2b",
  background: "#141414",
  color: "#f4f4f4",
  cursor: "pointer",
  fontWeight: 600,
};

export default App;