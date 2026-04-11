"""
CrossBorder Copilot — entry point
Run with:  python main.py
       or: uvicorn main:app --host 0.0.0.0 --port 8000
"""
from dotenv import load_dotenv
load_dotenv()
import uvicorn
from app.factory import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
