from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from .config import DATA_DIR

DB_PATH = DATA_DIR / "monitor.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL, PRIMARY KEY (symbol, date));
CREATE TABLE IF NOT EXISTS fng (date TEXT PRIMARY KEY, score REAL, rating TEXT, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS reports (date TEXT PRIMARY KEY, created_at TEXT, signal TEXT, target TEXT,
                                    data_json TEXT, html TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, created_at TEXT,
                                   kind TEXT, subject TEXT, body TEXT, emailed INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT, started_at TEXT,
                                 finished_at TEXT, status TEXT, message TEXT);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conn():
    """Connection that commits on success, rolls back on error, and always closes."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def init() -> None:
    with conn() as c:
        c.executescript(SCHEMA)


# ---- prices -------------------------------------------------------------
def upsert_prices(symbol: str, series: pd.Series) -> int:
    rows = [(symbol, pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)) for d, v in series.dropna().items()]
    with conn() as c:
        c.executemany("INSERT INTO prices(symbol,date,close) VALUES(?,?,?) "
                      "ON CONFLICT(symbol,date) DO UPDATE SET close=excluded.close", rows)
    return len(rows)


def load_prices(symbol: str) -> pd.Series:
    with conn() as c:
        rows = c.execute("SELECT date, close FROM prices WHERE symbol=? ORDER BY date", (symbol,)).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series([r["close"] for r in rows], index=pd.to_datetime([r["date"] for r in rows]), dtype=float)


# ---- fear & greed -------------------------------------------------------
def upsert_fng(rows: dict) -> int:
    """rows: {'YYYY-MM-DD': (score, rating)}"""
    ts = now_iso()
    with conn() as c:
        c.executemany("INSERT INTO fng(date,score,rating,fetched_at) VALUES(?,?,?,?) "
                      "ON CONFLICT(date) DO UPDATE SET score=excluded.score, rating=excluded.rating, "
                      "fetched_at=excluded.fetched_at",
                      [(d, float(s), r, ts) for d, (s, r) in rows.items()])
    return len(rows)


def load_fng() -> pd.Series:
    with conn() as c:
        rows = c.execute("SELECT date, score FROM fng ORDER BY date").fetchall()
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series([r["score"] for r in rows], index=pd.to_datetime([r["date"] for r in rows]), dtype=float)


# ---- reports ------------------------------------------------------------
def save_report(date: str, signal: str, target: str, data: dict, html: str, text: str) -> None:
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO reports(date,created_at,signal,target,data_json,html,text) "
                  "VALUES(?,?,?,?,?,?,?)", (date, now_iso(), signal, target, json.dumps(data, default=str), html, text))


def get_report(date: str | None = None):
    with conn() as c:
        if date:
            return c.execute("SELECT * FROM reports WHERE date=?", (date,)).fetchone()
        return c.execute("SELECT * FROM reports ORDER BY date DESC LIMIT 1").fetchone()


def list_reports(n: int = 60):
    with conn() as c:
        return c.execute("SELECT date, created_at, signal, target FROM reports ORDER BY date DESC LIMIT ?",
                         (n,)).fetchall()


# ---- alerts -------------------------------------------------------------
def add_alert(key: str, kind: str, subject: str, body: str) -> bool:
    """Insert alert; returns False if this key was already alerted (de-duplication)."""
    with conn() as c:
        cur = c.execute("INSERT OR IGNORE INTO alerts(key,created_at,kind,subject,body) VALUES(?,?,?,?,?)",
                        (key, now_iso(), kind, subject, body))
        return cur.rowcount == 1


def mark_alerts_emailed(keys: list[str]) -> None:
    with conn() as c:
        c.executemany("UPDATE alerts SET emailed=1 WHERE key=?", [(k,) for k in keys])


def list_alerts(n: int = 100):
    with conn() as c:
        return c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (n,)).fetchall()


# ---- runs ---------------------------------------------------------------
def start_run(job: str) -> int:
    with conn() as c:
        return c.execute("INSERT INTO runs(job,started_at,status) VALUES(?,?,'running')", (job, now_iso())).lastrowid


def finish_run(run_id: int, status: str, message: str = "") -> None:
    with conn() as c:
        c.execute("UPDATE runs SET finished_at=?, status=?, message=? WHERE id=?",
                  (now_iso(), status, message[:2000], run_id))


def list_runs(n: int = 30):
    with conn() as c:
        return c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (n,)).fetchall()
