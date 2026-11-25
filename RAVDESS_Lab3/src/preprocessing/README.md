# Preprocessing module

`audio.py` uses metadata parsing and actor pairing rather than ZIP ordering, reads WAV files into memory without filesystem extraction, checks sampling rates, and truncates all groups consistently.

Functions: `parse_filename`, `choose_matched_members`, `read_wav_in_memory`, `load_paired_audios`, `truncate_matched_groups`, `load_members_via_disk` (baseline).

`complexity_analysis.ipynb` contains a line-by-line source-code audit, whole-function bounds, and the rationale for eliminating intermediate filesystem writes.

For real RAVDESS, the default comparator selects both emotions using one identical actor, intensity, statement, and repetition. Equal truncation does not imply equal phonetic alignment.
