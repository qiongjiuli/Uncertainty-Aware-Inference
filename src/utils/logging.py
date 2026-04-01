"""
Shared logging and W&B utilities.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Optional[str | Path] = None, level: int = logging.INFO) -> None:
    """Configure root logger with stream + optional file handler."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / "run.log"))

    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers= handlers,
    )


def init_wandb(
    project : str = "uncertainty-aware-inference",
    name    : str = "run",
    tags    : Optional[list[str]] = None,
    config  : Optional[dict] = None,
):
    """Initialize W&B run. Returns run object or None if wandb unavailable."""
    try:
        import wandb
        run = wandb.init(
            project = project,
            name    = name,
            tags    = tags or [],
            config  = config or {},
            reinit  = True,
        )
        return run
    except ImportError:
        logging.getLogger(__name__).warning(
            "wandb not installed — skipping W&B logging. "
            "Install with: pip install wandb"
        )
        return None


def log_calibration(run, result, step: Optional[int] = None) -> None:
    if run is None:
        return
    from dataclasses import asdict
    d = asdict(result) if hasattr(result, "__dataclass_fields__") else dict(result)
    numeric = {k: v for k, v in d.items() if isinstance(v, (int, float))}
    run.log({"calibration/" + k: v for k, v in numeric.items()}, step=step)


def log_profiling(run, result, step: Optional[int] = None) -> None:
    if run is None:
        return
    from dataclasses import asdict
    d = asdict(result) if hasattr(result, "__dataclass_fields__") else dict(result)
    numeric = {k: v for k, v in d.items() if isinstance(v, (int, float))}
    run.log({"profiling/" + k: v for k, v in numeric.items()}, step=step)
