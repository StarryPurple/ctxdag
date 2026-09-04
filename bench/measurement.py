"""Shared, versioned report helpers for ContextDAG measurements."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "contextdag.measurement.v1"


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def revision_metadata(repo_root: Path | None = None) -> dict[str, Any]:
    """Describe the code under test without making Git a hard dependency."""
    root = repo_root or Path(__file__).resolve().parents[1]
    revision = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--porcelain", "--untracked-files=no"], root)
    return {
        "commit": revision,
        "dirty": bool(status) if status is not None else None,
    }


def new_report(
    experiment: str,
    backend: str,
    config: dict[str, Any],
    *,
    model: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Create the common envelope used by every formal benchmark."""
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "revision": revision_metadata(repo_root),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "backend": {"type": backend, "model": model},
        "config": config,
        "observations": [],
        "aggregates": {},
    }


def mean(values: Iterable[float | int]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def save_report(report: dict[str, Any], path: str | Path) -> None:
    """Atomically replace a report so interrupted runs remain valid JSON."""
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def session_counters(session: Any) -> dict[str, int]:
    """Snapshot protocol counters using stable report field names."""
    return {
        "expands": session.expands,
        "page_faults": session.page_faults,
        "requires_issued": session.requires_issued,
        "requires_rejected": session.requires_rejected,
        "catalog_chars": session.catalog_chars,
    }
