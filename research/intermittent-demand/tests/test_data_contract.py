from pathlib import Path

import pytest

from phase9.data_contract import Phase9Paths, create_run_dir, validate_snapshot


def test_validate_snapshot_requires_gold_silver_and_qfo(tmp_path: Path) -> None:
    paths = Phase9Paths.from_root(tmp_path)

    with pytest.raises(FileNotFoundError, match="gold"):
        validate_snapshot(paths)


def test_create_run_dir_refuses_to_overwrite(tmp_path: Path) -> None:
    expected = tmp_path / "artifacts" / "runs" / "trial"

    assert create_run_dir(tmp_path, "trial") == expected
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "trial")
