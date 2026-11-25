# Lab 3 — Modular acoustic analysis of acted neutral and sad speech

**Computational Foundations for Biomedical Engineering · Third Lab**  
**Language:** English. **Input:** RAVDESS *audio-only speech* ZIP or the supplied offline synthetic fixture. **Outputs:** feature tables, paired statistics, held-out-actor classification, plots, and benchmark results.

## Project goal and scientific scope

This project refactors the code from `Submission_2` into an auditable research pipeline with three modules: preprocessing, analysis, and visualization. It studies *acted neutral versus sad emotional speech*, not diagnoses of depression. The RAVDESS actors are not clinical patients. Claims about depression screening require clinically labeled data and external validation.

### Clinical and technical context

Depression has measurable behavioral correlates, including aspects of vocal expression, motivating research into automated assessment. The RAVDESS corpus provides controlled emotionally performed speech, suitable for studying acoustic analysis methods but not for estimating clinical screening accuracy (Livingstone & Russo, 2018; Mao et al., 2023).

### Directory layout

```text
RAVDESS_Lab3/
├── data/
│   ├── raw/synthetic_demo.zip       # Bundled, offline, generated signals (NOT RAVDESS)
│   └── README.md
├── src/
│   ├── preprocessing/audio.py      # File parsing, matching, ZIP I/O, equal length
│   ├── preprocessing/complexity_analysis.ipynb
│   ├── analysis/features.py        # Baseline loops and vectorized extraction
│   ├── analysis/statistics.py      # Original-style vs memory-oriented summaries, paired tests
│   ├── analysis/classification.py  # Leave-one-actor-out logistic models
│   ├── analysis/complexity_analysis.ipynb
│   ├── visualization/plots.py      # Paired plots and ROC visualization
│   └── visualization/complexity_analysis.ipynb
├── notebooks/
│   ├── Lab3_report.ipynb           # Main executable, self-contained report
│   └── Submission_2_corrected.ipynb
├── results/
│   ├── figures/                   # Generated figures
│   └── tables/                    # Generated features, tests, benchmark
├── docs/complexity_summary.md
├── tests/test_pipeline.py
├── benchmark.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

Each of the three module directories has its own README, **docstrings**, and executable notebook documenting **per-line and global complexity**. The report notebook is organized for detailed explanation; production algorithms are kept in scripts.

## Environment and one-command execution

Requires Python 3.11+ and pip. From the project root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py
python benchmark.py
python -m pytest -q
jupyter lab notebooks/Lab3_report.ipynb
```

The default dataset is a **small synthetic demonstration**, included in the archive. These files use RAVDESS-compatible filename codes solely for automated tests; they are not authentic emotional recordings and cannot support scientific conclusions about emotion recognition. This makes the repository testable offline without Colab `/content/` paths or registration.

### Run with actual RAVDESS data

1. Download `Audio_Speech_Actors_01-24.zip` from the [RAVDESS Zenodo record](https://zenodo.org/records/1188976) and keep the ZIP file intact.
2. Put it at `data/raw/Audio_Speech_Actors_01-24.zip` (not committed to Git because of size).
3. Run:

```bash
python run_pipeline.py --zip data/raw/Audio_Speech_Actors_01-24.zip --actors 24
python benchmark.py --zip data/raw/Audio_Speech_Actors_01-24.zip --actors 6
```

RAVDESS naming conventions are parsed by metadata: `03` audio-only, `01` speech, `01` neutral versus `04` sad, with matching actor, statement `01`, repetition `01`, and normal intensity `01`. The algorithm **does not** assume a fixed list position (such as `feature + 60*i`), which could select incorrect emotions. Both groups are globally truncated to the shortest included recording to give equal sample counts. This loses trailing audio and should be described as a methodological limitation.

## Methods

1. **Preprocessing** — Read WAVs in memory from ZIP archives, validate matching actors and sampling rate, align global recording lengths by truncation.
2. **Feature extraction** — 25 ms frames, 10 ms hop; RMS, Peak, ZCR, and Hann-windowed spectral centroid. pYIN estimates frame-level F0 with a longer analysis window selected to accommodate 65 Hz; unvoiced estimates remain missing.
3. **Data handling** — A table with one row per actor and a two-level column index `(Characteristic, Statistic)`; mean, sample standard deviation, median, minimum, maximum, and range.
4. **Statistics** — Paired Shapiro–Wilk and Wilcoxon signed-rank tests for each `(Characteristic, Statistic)` tuple; Holm correction for **multiple testing**. Paired actors are the analysis unit.
5. **Separability** — Logistic regression models, with **leave-one-actor-out** cross-validation. Both conditions from each held-out actor stay in the held-out fold; imputation and scaling are learned only from training folds. The feature subset is defined **before** examining results to reduce leakage.
6. **Visualizations** — Paired-condition plots and ROC curves saved under `results/figures/`.

### Baseline and optimizations

- **Ingestion:** ZIP `extract()` to disk (reference) versus reading compressed entries directly into memory (optimized). Both have linear I/O in sample count; the latter removes intermediate filesystem writes.
- **Feature computation:** Python loops across every actor/window (reference, included solely to quantify vectorization) versus the **already vectorized strategy present in Submission 2** (optimized). Both have `O(A·W·K log K)` for window-based spectral computations, but vectorization eliminates many Python-level loop operations.
- **Statistical summaries:** The previous notebook's `pd.concat` + `unstack` approach versus constructing one feature table at a time. The latter reduces temporary stacked-DataFrame size but can be **slower on small data**; benchmark results explicitly report that trade-off.
- **F0:** Both reference and optimized versions use the same pYIN implementation and parameters. pYIN is kept out of the feature-only benchmark to avoid hiding speed differences behind an unchanged, relatively expensive algorithm.

### Results and interpretation

Running the provided demo generates `neutral_features.csv`, `sad_features.csv`, `paired_statistics.csv`, `classification.csv`, and `cross_validated_predictions.csv`. It also writes separate paired feature plots and an actor-held-out ROC plot. These are **software-validation results only**: the demonstration audio consists of deterministic synthetic tones. It must not be interpreted as recognition accuracy for real actors or as evidence about depression.

The benchmark CSV reports observed reference and optimized times. Timing depends on CPU, filesystem, input duration, and cache/JIT state. **Do not claim asymptotic improvements from these timings:** both implementations remain dominated by spectral transforms and unchanged pYIN complexity. See `docs/complexity_summary.md` and the three complexity notebooks for symbolic derivations.

### Short discussion

The pipeline now enforces metadata-matched conditions, actor grouping, complete documentation, and reproducible tests. Trade-offs remain: truncation discards trailing information, the minimum F0 range is a modeling choice, silence remains included for amplitude/spectral features, multiple comparisons weaken exploratory significance, and acted emotions are not a depression reference standard.

### Conclusions

The Lab 3 pipeline separates responsibilities, provides explicit theoretical complexity bounds and empirical speed tests, and produces reproducible statistical/visual outputs. Its results establish reproducibility of the **method**, not clinical validity. Evaluate the complete RAVDESS ZIP before interpreting real-emotion separability.

## Git publication

From the extracted project directory, after checking which data may be shared:

```bash
git init
git add .
git commit -m "Lab 3: modular RAVDESS analysis and complexity comparison"
git branch -M main
git remote add origin <YOUR_GITHUB_OR_GITLAB_REPO_URL>
git push -u origin main
```

The repository has not been uploaded automatically. Do not commit the full RAVDESS archive without checking the dataset's redistribution conditions. The bundled synthetic fixture is expressly distributable for this demonstration.

## References (APA 7)

Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLOS ONE, 13*(5), Article e0196391. https://doi.org/10.1371/journal.pone.0196391

Mao, K., Wu, Y., & Chen, J. (2023). A systematic review on automated clinical depression diagnosis. *npj Mental Health Research, 2*, Article 20. https://doi.org/10.1038/s44184-023-00040-z
