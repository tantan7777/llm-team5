import { apiFetch } from "./config";

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

export async function sendMessage(
  apiBase: string,
  query: string,
  session_id?: string,
): Promise<ChatResponse> {
  return apiFetch<ChatResponse>(apiBase, "/chat/invoke", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      session_id: session_id ?? "",
    }),
  });
}

export async function getHistory(
  apiBase: string,
  session_id: string,
): Promise<HistoryResponse> {
  return apiFetch<HistoryResponse>(
    apiBase,
    `/chat/history/${encodeURIComponent(session_id)}`,
  );
}
