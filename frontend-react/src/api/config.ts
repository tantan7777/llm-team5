export const DEFAULT_API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const API_BASE_STORAGE_KEY = "crossborder_api_base";

export type HealthResponse = {
  service: string;
  status: string;
  chat_ready: boolean;
  llm_configured: boolean;
  database_path?: string;
  startup_error?: string | null;
};

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function normalizeApiBase(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return DEFAULT_API_BASE;
  return trimmed.replace(/\/+$/, "");
}

export function getStoredApiBase(): string {
  return normalizeApiBase(
    window.localStorage.getItem(API_BASE_STORAGE_KEY) ?? DEFAULT_API_BASE,
  );
}

export function storeApiBase(value: string): string {
  const normalized = normalizeApiBase(value);
  window.localStorage.setItem(API_BASE_STORAGE_KEY, normalized);
  return normalized;
}

export function buildApiUrl(apiBase: string, path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${normalizeApiBase(apiBase)}${normalizedPath}`;
}

async function parseError(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    const payload = await response.json().catch(() => null);
    if (payload?.detail) {
      return typeof payload.detail === "string"
        ? payload.detail
        : JSON.stringify(payload.detail);
    }
  }
  const text = await response.text().catch(() => "");
  return text || `Request failed with HTTP ${response.status}`;
}

export async function apiFetch<T>(
  apiBase: string,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(buildApiUrl(apiBase, path), init);
  if (!response.ok) {
    throw new ApiError(response.status, await parseError(response));
  }
  return response.json() as Promise<T>;
}

export function getHealth(apiBase: string): Promise<HealthResponse> {
  return apiFetch<HealthResponse>(apiBase, "/health");
}
