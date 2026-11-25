# Generated results

Run `python run_pipeline.py` and `python benchmark.py` from the root to refresh the generated tables and figures.

- `tables/neutral_features.csv`, `sad_features.csv` — actor rows, two-tier `Characteristic` and `Statistic` column headers.
- `tables/paired_statistics.csv` — Shapiro, Wilcoxon, and Holm-adjusted p-values for actor pairs.
- `tables/classification.csv` — leave-one-actor-out metric summaries.
- `tables/cross_validated_predictions.csv` — fold-held-out predictions and probabilities.
- `tables/benchmark.csv` — observed per-stage reference versus optimized timings (elapsed wall-clock seconds).
- `figures/paired_*.png` — separate actor-paired comparisons for preselected feature/statistic pairs.
- `figures/benchmark_comparison.png` — side-by-side timing of reference and optimized stages.
- `figures/roc_grouped_cv.png` — ROC curves constructed from actor-held-out probabilities.

The checked-in / supplied result files were generated from the **synthetic demo**, not RAVDESS. They are not scientific performance measurements.
