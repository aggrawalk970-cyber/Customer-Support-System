"""
app/db/connection.py
=====================
Backward-compatibility shim.

The canonical DB session is now in app.db.session.
This file re-exports everything so existing imports don't break during the transition.
"""
from app.db.session import Base, engine, SessionLocal, get_db  # noqa: F401
