"""One-time network fetch that freezes the ETT dataset as local Parquet snapshots.

Run by hand whenever the frozen snapshot needs to be (re)created:

    python scripts/fetch_ett.py

This script is the ONLY place in the ETT pipeline allowed to touch the network.
`datasets/ett.py` never imports `requests` and reads exclusively from the Parquet
files this script produces, so freezing here is what makes every later run
reproducible regardless of what the upstream mirror does afterwards.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "datasets" / "snapshots" / "ett"
MANIFEST_PATH = REPO_ROOT / "datasets" / "snapshots" / "MANIFEST.json"

SOURCE_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{variant}.csv"
)
VARIANTS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def fetch_variant(variant: str) -> dict:
    url = SOURCE_URL_TEMPLATE.format(variant=variant)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    csv_path = SNAPSHOT_DIR / f"{variant}.csv.tmp"
    csv_path.write_bytes(response.content)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    csv_path.unlink()

    parquet_path = SNAPSHOT_DIR / f"{variant}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    return {
        "sha256": _sha256_file(parquet_path),
        "row_count": int(len(df)),
        "date_range": [
            df["date"].min().isoformat(),
            df["date"].max().isoformat(),
        ],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_url": url,
    }


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    manifest.setdefault("ett", {})

    for variant in VARIANTS:
        print(f"fetching {variant} ...")
        manifest["ett"][variant] = fetch_variant(variant)
        print(f"  -> {manifest['ett'][variant]['row_count']} rows, "
              f"sha256={manifest['ett'][variant]['sha256'][:12]}...")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
