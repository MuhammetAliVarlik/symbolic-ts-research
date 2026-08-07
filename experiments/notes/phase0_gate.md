# Phase 0 gate — synthesis and decision

This is the explicit call the working plan's Phase 0 gate requires, synthesized across
F0-01 through F0-10. It states a decision, not another description of mixed evidence.

## Decision: **proceed to Phase 1, with two named caveats.** Not a hold, not an
## unconditional pass.

## The gate's two literal criteria

1. *"Transition matrices of the two domains are measurably more similar than chance"*
   (F0-07) — **fails at native granularity, and the granularity-matched version doesn't
   comfortably pass either.** The exact residual gap depends on which statistic and
   which daily-aggregation method is used (F0-11 checked this directly, not assumed):
   **Frobenius norm** — the more trustworthy of the two statistics, per the white-noise
   check below — shows a real gap under *both* `.last()` (96.8th percentile, 1.95 null
   sd) and `.mean()` (100th percentile, 6.74 null sd) daily resampling, actually larger
   under `.mean()`. **Symmetrised KL is aggregation-dependent** (80.4th percentile under
   `.last()`, 41.2nd — essentially the null's centre — under `.mean()`) and should not
   be cited as strong evidence in either direction going forward. The white-noise sanity
   check — run twice, once on the full 25-symbol vocabulary and again on change-only
   5-symbol tokens specifically to rule out the volatility channel's rolling-window
   artifact — confirms the granularity-matched similarity is **real, non-artifactual
   shared structure**, not a sigma-binning byproduct: real finance-vs-ett_daily sits at
   the 0th percentile of every noise-vs-real distribution on change-only tokens, on both
   Frobenius norm and symmetrised KL.
2. *"Cross-domain Markov performs clearly worse than in-domain Markov"* (F0-10) —
   **partially true, precisely characterized rather than left as an average.** The
   equalized-N control (subsampling finance's training set to ett_daily's size) shows
   the orders 2–3 portion of the original asymmetry was a training-set-size artifact
   (the gap closes under equal training data), while the orders 4–5 portion survives
   equalization — a genuine, sample-size-independent cross-domain penalty, but narrower
   than first reported, and asymmetric (only in the `ett_daily→finance` direction;
   `finance→ett_daily` shows no drop at any order, equalized or not).

**Neither criterion is a clean, textbook pass.** Both show a real signal underneath an
initially-alarming or initially-oversized number, once the actual confounds
(sampling-frequency mismatch, training-set-size mismatch) were checked directly instead
of assumed away. That pattern — a real effect, smaller and more precisely located than
the first measurement suggested — is why this is a "proceed with caveats" call, not an
unconditional pass.

## Supporting evidence across F0-01 through F0-10

- **F0-01–F0-05**: both domains frozen and reproducible; stationarity confirms
  differencing as the right transform for both (validating D2's change/volatility
  split, independently derived, before it was revised to drop `level`); token vocabulary
  usage is comparable across domains (F0-05), not degenerate in either.
- **F0-06**: each domain has substantial *individual* conditional structure (lag-1
  mutual information 1.74 bits ETT, 1.50 bits finance — 41–47% of each domain's marginal
  entropy) — there is something real in each domain for a model to learn, independent of
  whether it transfers.
- **F0-08**: entropy rate sits at 40–52% of the theoretical maximum in every domain and
  granularity tested — ruled out "sequences are close to random," the specific failure
  mode this task existed to check. The finance/ett_daily mutual-information *decay
  curves* converge closely (0.2–0.6% relative at several lags) once granularity is
  matched — a second, independent statistic (decay shape, not matrix distance) pointing
  at the same granularity-confound explanation F0-07 found.
- **F0-09**: ETT has a real, exact 24-hour seasonal component (measured precisely: peaks
  at lags 24 and 96 in the hourly and 15-min series respectively) that a context window
  cannot resolve regardless of length — this is a design implication for event tokens
  (D3), not a mark against the transfer hypothesis. Confirmed the volatility channel's
  known rolling-window artifact does not explain the long-range correlation that sets
  the (now PACF-informed, revised-down) context window of 50.
- **F0-10 + its equalized-N control**: see above — the clearest, most surprising result
  of Phase 0, precisely because the control could have gone either way and didn't
  confirm the simpler story in either direction.

## The two caveats carried forward, explicitly (not silently absorbed)

1. **The residual F0-07 gap on Frobenius norm (96.8th percentile under `.last()`,
   100th under `.mean()` — real under both, per F0-11) is unresolved, not explained
   away.** It should be re-checked once Phase 1's real walk-forward splitting (F1-06)
   and library-grade binning (F1-02/F1-03) exist — this Phase 0 work used a 70/30
   ad-hoc split and the throwaway F0-04 prototype tokenizer, not the eventual library.
   F1-11 should run that re-check under *both* daily-aggregation methods and report
   both, per F0-11's finding that the choice materially affects the result — not settle
   on a single number the way this phase's exploratory work did. A result that survives
   re-measurement with the real implementation is worth more than one that only exists
   in the prototype.
2. **The genuine orders 4–5 cross-domain Markov penalty is a real, standing bar, not a
   caveat against proceeding — but it should shape what "success" means for the SLM.**
   Since it survived a fair, sample-size-matched comparison, it is not something more
   classical-baseline data would fix; closing it requires a model that can exploit
   structure a higher-order lookup table cannot (longer-range/nonlinear structure —
   consistent with F0-08's still-substantial lag-10 MI and F0-09's short-PACF-but-
   long-ACF finding). If the eventual SLM does *not* close this specific, now-precisely-
   located gap, that is informative evidence against the thesis's transfer claim in a
   way a vaguer "cross-domain is worse" finding could not have supported as clearly.

Also carried forward but explicitly **not** a gate blocker: F0-07's deferred mean-based
resampling variant (untested; the `.last()` daily-resampling choice was never compared
against a daily-mean alternative) — a robustness check worth running before the thesis
write-up claims the granularity-matching result is robust to that specific choice, but
not a reason to hold Phase 1.

## Why this is "proceed," not "hold"

A hold would be justified if either gate criterion had failed with no explanation, or if
the white-noise/equalized-N controls had confirmed the alarming readings were entirely
artifactual with no real signal underneath. Neither happened. Both controls found a
smaller, more precisely located real effect than the raw numbers first suggested — which
is a *stronger* basis for proceeding than a naive pass would have been, because the
remaining claims (real shared structure exists; a real, order- and direction-specific
transfer penalty exists) are now backed by mechanism-level checks, not just a single
aggregate statistic. Proceeding without checking would have risked building Phase 1 on
an artifact; proceeding after checking and finding a real, smaller effect is exactly what
"the transfer hypothesis is real but not trivial" should look like at this stage.
