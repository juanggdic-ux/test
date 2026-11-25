# Analysis module

- `features.py`: reference scalar loops and vectorized 25 ms framing / 10 ms hop; RMS, Peak, ZCR, Hann-weighted spectral centroid, pYIN F0 using a longer window.
- `statistics.py`: original DataFrame stacking and memory-conscious per-feature summary, paired Shapiro / Wilcoxon tests and Holm multiple-comparison adjustment.
- `classification.py`: preselected feature subset and actor-held-out logistic classification, preventing leakage between the same actor's two emotional recordings.

All functions have docstrings. `complexity_analysis.ipynb` details algorithmic costs per line and per function, including an explicit opaque cost term for pYIN (a library routine whose internal complexity depends on implementation and parameters). Neither statistical significance nor synthetic test performance demonstrates depression classification.
