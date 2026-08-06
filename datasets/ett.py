"""Loader for the frozen ETT dataset snapshot.

Reads exclusively from the local Parquet snapshot under
``datasets/snapshots/ett/``. Performs no network I/O — to refresh the
snapshot, run ``scripts/fetch_ett.py`` by hand.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots" / "ett"

VALID_VARIANTS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")


def load_ett(variant: str) -> pd.DataFrame:
    """Load a frozen ETT variant from the local snapshot.

    Args:
        variant: One of "ETTh1", "ETTh2", "ETTm1", "ETTm2".

    Returns:
        DataFrame with columns ``date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT``.

    Raises:
        ValueError: If ``variant`` is not one of the known ETT variants.
        FileNotFoundError: If the snapshot has not been frozen yet.
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unknown ETT variant {variant!r}; expected one of {VALID_VARIANTS}"
        )

    path = _SNAPSHOT_DIR / f"{variant}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No frozen snapshot at {path}. Run scripts/fetch_ett.py to create it."
        )

    return pd.read_parquet(path, engine="pyarrow")
