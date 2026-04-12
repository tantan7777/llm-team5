export async function runEvaluation() {
  const res = await fetch("http://localhost:8000/evaluation/run", {
    method: "POST",
  });

  if (!res.ok) {
    throw new Error("Evaluation failed");
  }

  return res.json();
}