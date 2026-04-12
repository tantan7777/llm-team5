const API_BASE = "http://localhost:8000";

export type ChatResponse = {
  session_id: string;
  response: string;
  tool_calls_made: string[];
};

export type HistoryMessage = {
  role: string;
  content: string;
};

export type HistoryResponse = {
  session_id: string;
  messages: HistoryMessage[];
};

export async function sendMessage(query: string, session_id?: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat/invoke`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      session_id: session_id ?? "",
    }),
  });

  if (!res.ok) {
    throw new Error("Request failed");
  }

  return res.json();
}

export async function getHistory(session_id: string): Promise<HistoryResponse> {
  const res = await fetch(`${API_BASE}/chat/history/${session_id}`);

  if (!res.ok) {
    throw new Error("History request failed");
  }

  return res.json();
}