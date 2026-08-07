# symbolic-ts-research

Research repository for the thesis *"Modelling Time Series as a Symbolic Language:
Small Language Models for Cross-Domain Transfer on Edge Hardware."*

Depends on [`symbolic-ts`](../symbolic-ts) (the core library) and feeds the applied
demonstration in [`FinwiseBackend`](../FinwiseBackend).

## Status

**Phase 0 (feasibility) complete.** Phase 1 (core library) in progress.

Phase 0 asked a single question before any model was trained: *does a symbolic
representation carry anything genuinely shared between two unrelated time-series
domains — financial returns and an industrial sensor dataset (ETT) — or would training
be spent chasing a resemblance that isn't really there?*

## Phase 0 verdict

**Proceed to Phase 1, with two named caveats.** Not an unconditional pass — every
finding below was checked for the specific ways it could have been an artifact, and two
real ones were found and separated from what remains a genuine signal.

Full reasoning: `experiments/notes/phase0_gate.md`.

### What was tested, in order

| Task | Question | Finding |
|---|---|---|
| F0-01/02 | Are the datasets frozen and reproducible? | Yes — snapshotted, hashed, manifest-tracked |
| F0-03 | Are the series stationary? | No at raw level (expected); first-differencing resolves both domains cleanly |
| F0-04 | Does a naive symbolic binning survive contact with real data? | No — required log-returns (not raw differences) for finance, log-scale volatility binning, and dropping a non-stationary `level` channel (D2 revised: 2 channels, not 3) |
| F0-05 | Do the two domains use a comparable token vocabulary? | Yes — small effect size (Cramér's V = 0.108), shared top tokens, no vocabulary fragmentation |
| F0-06 | Does either domain have real sequential structure at all? | Yes, substantially — conditional structure accounts for ~41–47% of each domain's marginal entropy |
| F0-07 | Are the two domains' transition dynamics more similar than chance? | Fails at native sampling rate — but that turned out to be a granularity mismatch (ETT hourly, finance daily), not a real domain difference; matching granularity closes 92–96% of the gap |
| F0-08 | Is either sequence close to random? | No — entropy rate sits at 40–52% of the theoretical maximum in every domain tested |
| F0-09 | How far back should the model actually look? | PACF-derived context window: **50** steps (not the original placeholder of 30, and not the ACF-only estimate of 150, which conflated genuine volatility clustering with a rolling-window statistical artifact) |
| F0-10 | Does a cross-domain model perform clearly worse than an in-domain one? | Partially — a real, sample-size-independent penalty exists, but only at higher Markov orders (4–5) and only in one direction (ETT→finance); the equalized-training-size control ruled out training-set size as the explanation for that portion |

### The two caveats carried into Phase 1

1. F0-07's residual gap (real domains still sit at the 96.8th/80.4th percentile of a
   permutation null, not comfortably inside it) needs re-checking once Phase 1's real
   binning and split implementation replace the Phase 0 throwaway prototype.
2. The genuine cross-domain penalty found in F0-10 is now a precise, falsifiable target
   for the eventual SLM: closing that specific higher-order, direction-specific gap is
   what "the transfer claim holds" should mean in Phase 4, not just "cross-domain was
   worse than in-domain" in the abstract.

### Design decisions this phase produced or revised

- **D2** revised from a 3-channel (level, change, volatility) to a 2-channel (change,
  volatility) projection — a non-stationary level doesn't fit fixed training-derived
  thresholds.
- Finance's `change` channel uses **log returns**, not raw differences — raw dollar
  differences don't hold up over a multi-year window for a growing-price asset.
- **Context window: 50** (was an arbitrary placeholder of 30), derived from PACF, not
  the much longer but partly artifact-inflated ACF decay.
- Volatility is binned on a **log scale**, with near-zero windows excluded from the
  sigma fit itself, not just floored with an epsilon.

## Reproducing Phase 0

Every claim above is generated from a script and a result file, not asserted by hand:

```
scripts/           run_stationarity.py, run_token_frequency.py,
                    run_transition_matrices.py, run_transition_similarity.py,
                    run_information_theory.py, run_acf_pacf.py,
                    run_markov_baseline.py, run_markov_equalized_n.py,
                    run_white_noise_sanity_check.py, run_white_noise_change_only.py
results/            *.csv outputs referenced in each report below
figures/output/      A1–A9 figure set, generated from results/*.csv
experiments/notes/    per-task reports (F0-01.md … F0-10.md) and phase0_gate.md
```

No number in this README was hand-typed from memory; each traces to one of the files
above.

## Repository layout

```
datasets/       finance and ETT loaders, frozen snapshots, MANIFEST.json
experiments/    baselines, configs, per-task notes
figures/        style module + generation scripts
results/        CSV outputs, one source of truth per figure/table
scripts/        entry points for every Phase 0 task
tests/          (Phase 1 onward)
```

## Related repositories

- [`symbolic-ts`](../symbolic-ts) — the core library this research depends on
- [`FinwiseBackend`](../FinwiseBackend) — the applied demonstration this research feeds
