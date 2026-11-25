"""Paths, input validation, and output isolation for Phase9 experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Phase9Paths:
    """The immutable Phase9 snapshot paths required by the mainline."""

    root: Path
    snapshot: Path
    gold: Path
    silver: Path
    qfo: Path
    manifest: Path

    @classmethod
    def from_root(cls, root: Path) -> "Phase9Paths":
        root = root.resolve()
        snapshot = root / "data" / "phase9_20260817"
        return cls(
            root=root,
            snapshot=snapshot,
            gold=snapshot / "gold" / "wide_table_sku.csv",
            silver=snapshot / "silver" / "clean_orders.csv",
            qfo=snapshot / "metadata" / "qfo_availability.csv",
            manifest=snapshot / "source_manifest.json",
        )


def validate_snapshot(paths: Phase9Paths) -> dict[str, object]:
    """Confirm that the frozen inputs needed for reproducible experiments exist."""

    required = {
        "gold": paths.gold,
        "silver": paths.silver,
        "qfo": paths.qfo,
        "manifest": paths.manifest,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase9 snapshot input(s): {', '.join(missing)}")
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def create_run_dir(project_root: Path, run_id: str) -> Path:
    """Create one non-overwritable directory for a new research run."""

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty directory name")
    path = project_root.resolve() / "artifacts" / "runs" / run_id
    if path.exists():
        raise FileExistsError(f"Run directory already exists: {path}")
    path.mkdir(parents=True)
    return path
