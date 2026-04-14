from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.database import get_connection


def save_query(data: dict[str, Any]) -> None:
    print("Saving to DB:", data)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO history (query, provider, trust_score, geo_score, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            data.get("query"),
            data.get("provider"),
            data.get("trust_score"),
            data.get("geo_score", 0),
            str(datetime.utcnow()),
        ),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 10) -> list[dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, query, provider, trust_score, geo_score, timestamp
        FROM history
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    print("Fetched rows:", rows)
    conn.close()

    return [
        {
            "id": r[0],
            "query": r[1],
            "provider": r[2],
            "trust_score": r[3],
            "geo_score": r[4],
            "timestamp": r[5],
        }
        for r in rows
    ]

