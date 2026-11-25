# Algorithmic complexity and measured optimization (Lab 3)

## Symbols and assumptions

- `A`: number of matched actors; `C = 2`: neutral and sad conditions; `R = C·A`: total recordings.
- `N`: samples per (truncated) recording; `Z`: archive entries; `L`: total raw audio bytes.
- `K`: samples in one short-time frame (25 ms); `H`: hop samples (10 ms); `W = 1 + floor((N−K)/H)` analysis windows per recording.
- `P`: pYIN frame samples (larger than K for low pitch); `W0 = 1 + floor((N−P)/H)` pitch windows.
- `F = 5`: acoustic characteristics; `S = 6`: summary statistics; `d = 4`: preselected classifier features; `M = 2`: classifiers.
- `T_pYIN(R,N,P,H)`: *actual* pYIN routine cost (implementation-dependent, so not falsely simplified to O(R·W0·P)).
- `T_fit(n,d)`: cost for fitting one classifier to n labeled rows and d features; depends on solver and iterations.
- Python loops and vectorized NumPy operations can share asymptotic complexity but have very different constant factors.

## Preprocessing

| Implementation | Time complexity | Auxiliary storage | Notes |
|---|---|---|---|
| Scan archive metadata and pair actors | `O(Z + A log A)` | `O(Z + A)` | One scan plus sort of matched actor keys; filename lengths treated bounded |
| Reference: extract ZIP to disk then decode | `O(L + R·N)` | `O(R·N)` disk + decoded arrays | Filesystem writes and subsequent reads |
| Optimized: ZIP bytes to `BytesIO` then decode | `O(L + R·N)` | `O(R·N)` RAM | Avoids intermediate extraction, same big-O |
| Truncate to one global shortest recording | `O(R·N)` | `O(R·N)` output | Finds min in O(R), copies selected prefixes |

## Feature extraction

| Stage | Time | Additional storage |
|---|---|---|
| Frame view and axis swap | `O(1)` view bookkeeping | `O(1)` excluding source |
| RMS, peak, ZCR | `O(R·W·K)` | `O(R·W)` outputs plus temporaries |
| Hann application | `O(R·W·K)` | `O(R·W·K)` weighted view copy |
| FFT for spectral centroid | `O(R·W·K log K)` | `O(R·W·K)` for spectra/temporaries |
| Magnitude-weighted spectral centroid | `O(R·W·K)` | `O(R·W)` output |
| Framewise F0 via pYIN | `T_pYIN(R,N,P,H)` | about `O(R·W0)` output plus library working storage |

**Reference loop and optimized vectorized feature extraction both have**

`T_features = O(R·W·K log K) + T_pYIN(R,N,P,H)`.

**But** reference loops have `O(R·W)` Python iterations. Vectorization moves most inner operations into optimized C/NumPy routines (constant-factor speedup). A vectorized array may require more peak memory; do not claim a memory reduction for this stage.

## Data analysis and inference

| Stage | Time | Auxiliary memory |
|---|---|---|
| Reference `pd.concat` stacked window DataFrame | `O(F·R·W)` | `O(F·R·W)` |
| Optimized one-feature-at-a-time summaries | `O(F·R·W)` | `O(R·max(W,W0) + F·R·S)` |
| Statistical aggregation (mean/std/min/max/median) | `O(F·R·W log W)` worst-case conservative (median may sort); often closer to linear with selection | `O(F·R·S)` result |
| Paired Shapiro and Wilcoxon for F·S measures | `O(F·S·A log A)` conservative sorting bound plus library-dependent test costs | `O(F·S·A)` |
| Holm correction for F·S p-values | `O(F·S log(F·S))` | `O(F·S)` |
| Leave-one-actor-out classifier | `O(M·A·T_fit(2A−2,d))` plus `O(M·A·d)` predictions | `O(A·d + M·A)` for fold arrays and results |

No global p-value feature selection is performed before cross-validation. Every test fold contains both emotion conditions of one held-out actor.

## Visualization

- Paired-feature plots: prepare `O(F_selected·A)` points, then render approximately proportional to marks and output pixels; Matplotlib backend cost is implementation-dependent.
- ROC curves: `O(M·R log R)` for threshold sorting, plus drawing/export overhead.
- See the module's `complexity_analysis.ipynb` for statement-by-statement details.

## Combined pipeline

The non-opaque numerical portion is bounded by

`O(Z + L + R·N + R·W·K log K + F·R·W log W + F·S·A log A + M·A·T_fit(2A−2,d))`

plus **`T_pYIN(R,N,P,H)`** and plot rendering. The spectrum and pitch estimator dominate analysis on moderate recordings. Disk I/O also matters for large archives.

## Computational timing comparisons

`python benchmark.py` writes `results/tables/benchmark.csv`. It times the same WAV content with identical frame sizes, and checks reference/optimized equality. The loop/array benchmark *excludes unchanged pYIN*, so performance differences are attributable to the operations under study. Results are measured on the bundled synthetic signals, not claims about RAVDESS speed on all hardware.

**Important nuance:** `summarize_optimized` is **memory-oriented**. On small datasets, the original `pd.concat` summary may run faster because Pandas executes fewer high-level calls; the timing table must be reported, not assumed.

## Correctness and scientific constraints

The synthetic fixture intentionally uses RAVDESS-like filenames but is not original RAVDESS audio. It is suitable for unit tests and timing only. Truncation can remove informative end-of-recording content, framewise F0 is undefined for unvoiced segments, p-values are exploratory with multiple-testing correction, and emotion classification is not depression screening.
