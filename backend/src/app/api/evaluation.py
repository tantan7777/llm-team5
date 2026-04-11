from pathlib import Path
import os
import sys

from fastapi import APIRouter

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from eval_retrieval import run_evaluation

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.post("/run")
async def run_eval():
    old_cwd = os.getcwd()

    try:
        os.chdir(ROOT_DIR)
        results = run_evaluation(verbose=False)
    finally:
        os.chdir(old_cwd)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    return {
        "total": total,
        "passed": passed,
        "accuracy": round((passed / total) * 100, 2) if total else 0.0,
        "results": results,
    }