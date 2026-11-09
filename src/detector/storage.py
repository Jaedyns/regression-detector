import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from detector.evaluate import CaseResult

DB_PATH = Path(__file__).resolve().parents[2] / "runs" / "results.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at     TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model          TEXT NOT NULL,
    repeats        INTEGER NOT NULL,
    git_sha        TEXT
);
CREATE TABLE IF NOT EXISTS results (
    run_id   INTEGER NOT NULL REFERENCES runs(id),
    case_id  TEXT NOT NULL,
    repeat   INTEGER NOT NULL,
    category TEXT,
    summary  TEXT,
    error    TEXT,
    checks   TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id, repeat)
);
"""


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=DB_PATH.parents[1]).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def save_run(results: list[CaseResult], prompt_version: str, model: str, repeats: int,
             path: Path = DB_PATH) -> int:
    with connect(path) as conn:
        run_id = conn.execute(
            "INSERT INTO runs (started_at, prompt_version, model, repeats, git_sha) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), prompt_version, model, repeats, git_sha()),
        ).lastrowid
        conn.executemany(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(run_id, r.case_id, r.repeat,
              r.output.category if r.output else None,
              r.output.summary if r.output else None,
              r.error,
              json.dumps([c.model_dump() for c in r.checks]))
             for r in results],
        )
    return run_id
