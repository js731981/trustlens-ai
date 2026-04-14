"""Ranking drift tracker.

Stores point-in-time (query, product, rank, timestamp) rows and computes drift metrics over time.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings

_LOCK = threading.Lock()
_DB_INITIALIZED = False


def _normalize_query(q: str) -> str:
    return " ".join((q or "").strip().lower().split())


def query_key(query: str) -> str:
    normalized = _normalize_query(query)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _db_path() -> Path:
    settings = get_settings()
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "trust_lens.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    """Create ranking_drift table if missing (idempotent)."""
    global _DB_INITIALIZED
    with _LOCK:
        if _DB_INITIALIZED:
            return
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ranking_drift (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    query_key TEXT NOT NULL,
                    product TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ranking_drift_qk_time
                    ON ranking_drift(query_key, datetime(timestamp) ASC);
                CREATE INDEX IF NOT EXISTS idx_ranking_drift_qk_product_time
                    ON ranking_drift(query_key, product, datetime(timestamp) ASC);
                """
            )
        finally:
            conn.close()
        _DB_INITIALIZED = True


def track_drift(query: str, rankings: list[dict[str, Any]]) -> None:
    """
    Persist a ranking snapshot as one row per product.

    Expected rankings items: {"name": str, "rank": int} (extra keys ignored).
    """
    q = (query or "").strip()
    if not q:
        return
    init_db()
    qk = query_key(q)
    ts = datetime.now(tz=UTC).isoformat()

    rows: list[tuple[str, str, str, int, str]] = []
    for item in rankings or []:
        if not isinstance(item, dict):
            continue
        product = str(item.get("name") or item.get("product") or "").strip()
        if not product:
            continue
        try:
            rank = int(item.get("rank"))
        except Exception:
            continue
        if rank <= 0:
            continue
        rows.append((q, qk, product, rank, ts))

    if not rows:
        return

    with _LOCK:
        conn = _connect()
        try:
            conn.executemany(
                """
                INSERT INTO ranking_drift (query, query_key, product, rank, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        finally:
            conn.close()


def get_drift(query: str, limit: int = 2000) -> tuple[list[dict[str, Any]], float]:
    """
    Return (history, drift_score).

    history items: {query, product, rank, timestamp, rank_change}
    drift_score: 0..1 volatility-like score derived from mean absolute rank changes.
    """
    q = (query or "").strip()
    if not q:
        return [], 0.0
    init_db()
    qk = query_key(q)
    n = int(limit) if int(limit) > 0 else 2000

    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT query, product, rank, timestamp
            FROM ranking_drift
            WHERE query_key = ?
            ORDER BY datetime(timestamp) ASC, product ASC, id ASC
            LIMIT ?
            """,
            (qk, n),
        ).fetchall()
    finally:
        conn.close()

    history: list[dict[str, Any]] = []
    last_rank_by_product: dict[str, int] = {}
    deltas: list[int] = []
    max_rank = 0

    for r in rows:
        product = str(r["product"])
        rank = int(r["rank"])
        max_rank = max(max_rank, rank)
        prev = last_rank_by_product.get(product)
        change = None if prev is None else (rank - prev)
        if prev is not None:
            deltas.append(abs(rank - prev))
        last_rank_by_product[product] = rank
        history.append(
            {
                "query": str(r["query"]),
                "product": product,
                "rank": rank,
                "timestamp": str(r["timestamp"]),
                "rank_change": change,
            }
        )

    if not deltas:
        return history, 0.0

    mean_abs = sum(deltas) / float(len(deltas))
    denom = float(max(1, max_rank - 1))
    score = mean_abs / denom
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return history, float(score)

