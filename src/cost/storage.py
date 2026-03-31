"""SQLite-based persistent storage for cost tracking data."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from src.cost.calculator import CostEstimate


class CostStorage:
    """Persistent cost event storage using SQLite."""

    def __init__(self, db_path: str = "data/cost_history.db") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the cost_events table if it doesn't exist."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    cost_if_large_usd REAL NOT NULL,
                    complexity_score REAL DEFAULT 0.0,
                    routing_decision TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON cost_events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model ON cost_events(model_name)
            """)

    def record(
        self,
        estimate: CostEstimate,
        complexity_score: float = 0.0,
        routing_decision: str = "",
    ) -> None:
        """Store a cost event."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO cost_events
                   (timestamp, model_name, prompt_tokens, completion_tokens,
                    cost_usd, cost_if_large_usd, complexity_score, routing_decision)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    estimate.model_name,
                    estimate.prompt_tokens,
                    estimate.completion_tokens,
                    estimate.cost_usd,
                    estimate.cost_if_large_model_usd,
                    complexity_score,
                    routing_decision,
                ),
            )

    def get_total_cost(self) -> float:
        """Get total accumulated cost."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM cost_events").fetchone()
            return row[0]

    def get_total_savings(self) -> float:
        """Get total savings vs always using large model."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_if_large_usd - cost_usd), 0) FROM cost_events"
            ).fetchone()
            return max(0.0, row[0])

    def get_request_count(self) -> int:
        """Get total number of recorded requests."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM cost_events").fetchone()
            return row[0]

    def get_recent(self, n: int = 50) -> list[dict]:
        """Get the most recent N cost events."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM cost_events ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
            return [dict(row) for row in rows]

    def clear(self) -> None:
        """Clear all stored data."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM cost_events")
