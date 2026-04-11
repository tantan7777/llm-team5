"""
app/tools/notepad.py — Notepad tool for persisting session-scoped key/value notes.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum

from langchain_core.tools import tool
from sqlmodel import Session, select

from app.db.database import NotepadEntry, engine


class NotepadAction(str, Enum):
    read  = "read"
    write = "write"


@tool
def notepad(action: str, session_id: str, key: str = "", value: str = "") -> str:
    """Persist or retrieve key/value notes scoped to a session.

    Args:
        action:     "read" or "write".
        session_id: The current conversation session ID.
        key:        The note label (e.g. "hs_code", "shipper_address").
        value:      The note content (required for 'write').

    Returns a JSON string with the result.
    """
    with Session(engine) as db:
        if action == NotepadAction.write:
            if not key or not value:
                return json.dumps({"error": "Both 'key' and 'value' are required for write."})
            existing = db.exec(
                select(NotepadEntry)
                .where(NotepadEntry.session_id == session_id)
                .where(NotepadEntry.key == key)
            ).first()
            if existing:
                existing.value      = value
                existing.updated_at = datetime.utcnow().isoformat()
                db.add(existing)
            else:
                db.add(NotepadEntry(session_id=session_id, key=key, value=value))
            db.commit()
            return json.dumps({"status": "saved", "key": key, "value": value})

        elif action == NotepadAction.read:
            if key:
                entry = db.exec(
                    select(NotepadEntry)
                    .where(NotepadEntry.session_id == session_id)
                    .where(NotepadEntry.key == key)
                ).first()
                if not entry:
                    return json.dumps({"error": f"No note found for key '{key}'."})
                return json.dumps({
                    "key":        entry.key,
                    "value":      entry.value,
                    "updated_at": entry.updated_at,
                })
            else:
                entries = db.exec(
                    select(NotepadEntry)
                    .where(NotepadEntry.session_id == session_id)
                ).all()
                return json.dumps([
                    {"key": e.key, "value": e.value, "updated_at": e.updated_at}
                    for e in entries
                ])

    return json.dumps({"error": f"Unknown action '{action}'. Use 'read' or 'write'."})
