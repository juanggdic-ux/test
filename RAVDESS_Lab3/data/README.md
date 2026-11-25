# Data sources and format

- `raw/synthetic_demo.zip`: six actor IDs, paired neutral/sad artificial sine-wave mixtures, 8 kHz. Generated *only* to test parsing, audio processing, statistics, plots, and timing without a network connection. These are **not real speech or RAVDESS records**.
- Optional real dataset: `Audio_Speech_Actors_01-24.zip` from https://zenodo.org/records/1188976 . Do not upload it to Git inadvertently.
- ZIP entries follow `03-01-{emotion}-01-01-01-{actor}.wav` for the small synthetic test. In authentic RAVDESS: audio-only = 03, speech = 01, neutral = 01, sad = 04.
- Original durations may differ. All selected recordings are truncated to a *single* common minimum sample count before calculating features. This prevents trailing padding from altering RMS and silence-related measures, at the cost of discarding trailing content.
- No patient diagnoses exist in this project; the identifiers are actor IDs.
