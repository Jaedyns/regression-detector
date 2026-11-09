import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from detector.classifier import Classification
from detector.evaluate import CaseResult
from detector.evaluators import CheckResult

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


class StoredRun(BaseModel):
    run_id: int
    started_at: str
    prompt_version: str
    model: str
    repeats: int
    git_sha: str | None
    results: list[CaseResult]


def load_run(run_id: int, path: Path = DB_PATH) -> StoredRun | None:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT started_at, prompt_version, model, repeats, git_sha FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        rows = conn.execute(
            "SELECT case_id, repeat, category, summary, error, checks FROM results WHERE run_id = ? "
            "ORDER BY repeat, case_id", (run_id,)
        ).fetchall()
    results = [
        CaseResult(
            case_id=case_id,
            repeat=repeat,
            output=Classification(category=category, summary=summary) if category is not None else None,
            error=error,
            checks=[CheckResult(**c) for c in json.loads(checks)],
        )
        for case_id, repeat, category, summary, error, checks in rows
    ]
    started_at, prompt_version, model, repeats, sha = row
    return StoredRun(run_id=run_id, started_at=started_at, prompt_version=prompt_version,
                     model=model, repeats=repeats, git_sha=sha, results=results)
