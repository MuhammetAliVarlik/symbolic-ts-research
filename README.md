# symbolic-ts-research

Thesis experiments for *Modelling Time Series as a Symbolic Language: Small Language
Models for Cross-Domain Transfer on Edge Hardware*.

## Datasets

Datasets are frozen locally under `datasets/snapshots/` and loaded from there only — no
network I/O happens at experiment time. See `datasets/snapshots/MANIFEST.json` for
provenance (source URL, SHA256, retrieval timestamp) of every frozen file.

To (re)freeze a dataset snapshot, run the corresponding script under `scripts/`, e.g.:

```bash
python scripts/fetch_ett.py
```
