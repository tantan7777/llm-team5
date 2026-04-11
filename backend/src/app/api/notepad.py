"""
app/api/notepad.py — /notepad routes.
"""

from __future__ import annotations

from fastapi import APIRouter
from sqlmodel import Session, select

from app.db.database import NotepadEntry, engine

router = APIRouter(tags=["Notepad"])


@router.get("/notepad/{session_id}")
async def get_notepad(session_id: str):
    """Directly inspect all notepad entries for a session."""
    with Session(engine) as db:
        entries = db.exec(
            select(NotepadEntry).where(NotepadEntry.session_id == session_id)
        ).all()
    return {
        "session_id": session_id,
        "notes":      [e.dict() for e in entries],
    }
