"""
app/db/database.py — SQLModel schema and DB initialisation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel, create_engine

from app.core.config import SQLITE_URL, DB_PATH

logger = logging.getLogger("crossborder-copilot")


class NotepadEntry(SQLModel, table=True):
    id:         Optional[int] = Field(default=None, primary_key=True)
    session_id: str           = Field(index=True)
    key:        str
    value:      str
    updated_at: str           = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


engine = create_engine(SQLITE_URL, echo=False)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    logger.info("SQLite DB ready at %s", DB_PATH)
