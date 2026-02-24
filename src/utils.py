import time
from pathlib import Path


def make_run_dir(output_root: Path) -> Path:
    """
    Create a new directory for the current run, named with the current timestamp.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir