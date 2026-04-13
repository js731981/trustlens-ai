"""Persistent logging: LLM outputs, trust score history, ranking drift vs prior same-query runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from app.core.config import get_settings
from app.models.analyze import HistoryEntry
from app.services.ranking_consistency import normalize_ranking_key

_LOCK = threading.Lock()
_DB_INITIALIZED = False


def query_key(user_query: str) -> str:
    normalized = " ".join(user_query.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _db_path() -> Path:
    settings = get_settings()
    path = settings.data_dir
    path.mkdir(parents=True, exist_ok=True)
    return path / "trust_lens.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    """Create tables if missing (idempotent). Call once at app startup."""
    global _DB_INITIALIZED
    with _LOCK:
        if _DB_INITIALIZED:
            return
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS llm_responses (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    run_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    template_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    query_key TEXT NOT NULL,
                    raw_content TEXT NOT NULL,
                    parsed_json TEXT,
                    parse_error TEXT,
                    model TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_llm_session ON llm_responses(session_id);
                CREATE INDEX IF NOT EXISTS idx_llm_query_key ON llm_responses(query_key);

                CREATE TABLE IF NOT EXISTS analyze_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    query_key TEXT NOT NULL,
                    trust_score REAL NOT NULL,
                    ranking_json TEXT NOT NULL,
                    drift_score REAL,
                    kendall_tau REAL,
                    prior_trust_score REAL,
                    prior_run_id TEXT,
                    prior_run_at TEXT,
                    n_items_drift INTEGER NOT NULL DEFAULT 0,
                    snapshot_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runs_query_time ON analyze_runs(query_key, created_at DESC);
                """,
            )
        finally:
            conn.close()
        _DB_INITIALIZED = True


def kendall_tau_rank_agreement(prev_order: list[str], curr_order: list[str]) -> tuple[float | None, int]:
    """
    Kendall's tau (-1..1) between two total orderings restricted to items present in both.
    Returns (tau, n_items) or (None, n_items) if n_items < 2.
    """
    pos_prev: dict[str, int] = {}
    for i, name in enumerate(prev_order):
        k = normalize_ranking_key(name)
        if k not in pos_prev:
            pos_prev[k] = i
    pos_curr: dict[str, int] = {}
    for i, name in enumerate(curr_order):
        k = normalize_ranking_key(name)
        if k not in pos_curr:
            pos_curr[k] = i
    keys = sorted(set(pos_prev) & set(pos_curr))
    n = len(keys)
    if n < 2:
        return (None, n)
    xa = [pos_prev[k] for k in keys]
    xb = [pos_curr[k] for k in keys]
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = 1 if xa[i] < xa[j] else (-1 if xa[i] > xa[j] else 0)
            sign_b = 1 if xb[i] < xb[j] else (-1 if xb[i] > xb[j] else 0)
            prod = sign_a * sign_b
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return (1.0, n)
    tau = (concordant - discordant) / total
    return (float(max(-1.0, min(1.0, tau))), n)


def drift_from_tau(tau: float | None) -> float | None:
    """Map tau (1 = identical) to 0..1 drift where 0 is no drift."""
    if tau is None:
        return None
    return float(max(0.0, min(1.0, (1.0 - tau) / 2.0)))


@dataclass(frozen=True)
class PriorRun:
    id: str
    trust_score: float
    created_at: datetime
    ranking_names: list[str]


def fetch_prior_run_for_query(query_key_value: str) -> PriorRun | None:
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT id, trust_score, created_at, ranking_json
            FROM analyze_runs
            WHERE query_key = ?
            ORDER BY datetime(created_at) DESC
            LIMIT 1
            """,
            (query_key_value,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    names = json.loads(row["ranking_json"])
    if not isinstance(names, list):
        return None
    return PriorRun(
        id=str(row["id"]),
        trust_score=float(row["trust_score"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        ranking_names=[str(x) for x in names],
    )


def record_llm_response(
    *,
    response_id: str,
    session_id: str,
    run_index: int,
    template_id: str,
    user_query: str,
    raw_content: str,
    parsed_json: str | None,
    parse_error: str | None,
    model: str,
) -> None:
    init_db()
    qk = query_key(user_query)
    created_at = datetime.now(tz=UTC).isoformat()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO llm_responses (
                    id, session_id, run_index, created_at, template_id, user_query, query_key,
                    raw_content, parsed_json, parse_error, model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response_id,
                    session_id,
                    run_index,
                    created_at,
                    template_id,
                    user_query,
                    qk,
                    raw_content,
                    parsed_json,
                    parse_error,
                    model,
                ),
            )
        finally:
            conn.close()


def record_analyze_run(
    *,
    run_id: str,
    user_query: str,
    trust_score: float,
    ranking_names: list[str],
    snapshot: dict[str, Any],
    drift_score: float | None,
    kendall_tau: float | None,
    prior_trust_score: float | None,
    prior_run_id: str | None,
    prior_run_at: datetime | None,
    n_items_drift: int,
) -> None:
    init_db()
    qk = query_key(user_query)
    created_at = datetime.now(tz=UTC).isoformat()
    ranking_json = json.dumps(ranking_names, ensure_ascii=False)
    snapshot_json = json.dumps(snapshot, ensure_ascii=False, default=str)
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO analyze_runs (
                    id, created_at, user_query, query_key, trust_score, ranking_json,
                    drift_score, kendall_tau, prior_trust_score, prior_run_id, prior_run_at,
                    n_items_drift, snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    user_query,
                    qk,
                    trust_score,
                    ranking_json,
                    drift_score,
                    kendall_tau,
                    prior_trust_score,
                    prior_run_id,
                    prior_run_at.isoformat() if prior_run_at else None,
                    n_items_drift,
                    snapshot_json,
                ),
            )
        finally:
            conn.close()


def list_analyze_history(limit: int = 200) -> list[HistoryEntry]:
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, user_query, trust_score, created_at, snapshot_json
            FROM analyze_runs
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    out: list[HistoryEntry] = []
    for row in rows:
        snap = json.loads(row["snapshot_json"])
        out.append(
            HistoryEntry(
                id=UUID(str(row["id"])),
                query=str(row["user_query"]),
                trust_score=float(row["trust_score"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                snapshot=snap if isinstance(snap, dict) else None,
            ),
        )
    return out
