import { apiFetch } from "./config";

export async function runEvaluation(apiBase: string) {
  return apiFetch(apiBase, "/evaluation/run", {
    method: "POST",
  });
}
