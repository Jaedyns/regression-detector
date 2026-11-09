from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from detector.stats import RunStats
from detector.storage import git_sha

BASELINE_PATH = Path(__file__).resolve().parents[2] / "baselines" / "baseline.json"


class Baseline(BaseModel):
    created_at: str
    git_sha: str | None
    run_id: int | None
    stats: RunStats


def save_baseline(stats: RunStats, run_id: int | None, sha: str | None = None,
                  path: Path = BASELINE_PATH) -> Baseline:
    baseline = Baseline(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=sha or git_sha(),
        run_id=run_id,
        stats=stats,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline.model_dump_json(indent=2) + "\n")
    return baseline


def load_baseline(path: Path = BASELINE_PATH) -> Baseline | None:
    return Baseline.model_validate_json(path.read_text()) if path.exists() else None
