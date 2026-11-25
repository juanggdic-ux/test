"""Timing comparison for equivalent reference and vectorized pipeline stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import json
import zipfile

import numpy as np
import pandas as pd

from src.preprocessing.audio import choose_matched_members, load_members_via_disk, read_wav_in_memory, truncate_matched_groups
from src.analysis.features import window_features_reference, window_features_vectorized
from src.analysis.statistics import summarize_reference, summarize_optimized
from src.visualization.plots import plot_benchmark_comparison

ROOT = Path(__file__).resolve().parent


def measure(fn, repeats: int = 4):
    """Return best observed execution time and result; repeat to reduce timer noise."""
    times, result = [], None
    for _ in range(repeats):
        start = perf_counter()
        result = fn()
        times.append(perf_counter() - start)
    return min(times), result


def benchmark(zip_path: str | Path, n_actors: int = 6, repeats: int = 4) -> pd.DataFrame:
    """Compare disk/memory ingestion, loop/vector features, and stats construction.

    F0 estimation is excluded from the *feature-timing* comparison because the
    identical pYIN routine is used in both implementations. See README.
    """
    with zipfile.ZipFile(zip_path) as z:
        actors, members = choose_matched_members(z.namelist(), n_actors)
        filenames = members["Neutral"]

    with TemporaryDirectory() as tmp:
        t_disk, read_disk = measure(lambda: load_members_via_disk(zip_path, filenames, tmp), repeats)
    def read_memory():
        with zipfile.ZipFile(zip_path) as archive:
            return [read_wav_in_memory(archive, name)[0] for name in filenames]
    t_mem, read_mem = measure(read_memory, repeats)
    if not all(np.allclose(a, b, atol=1e-6) for a, b in zip(read_disk, read_mem)):
        raise AssertionError("Disk and in-memory WAV loaders disagree")

    matrix = truncate_matched_groups({"Neutral": read_mem})["Neutral"]
    # First call warms up library JIT work; neither baseline nor optimized pays it.
    window_features_reference(matrix[:1], int(_sampling_rate(zip_path, filenames[0])), include_pitch=False)
    fs = _sampling_rate(zip_path, filenames[0])
    t_loop, reference = measure(lambda: window_features_reference(matrix, fs, include_pitch=False), repeats)
    t_vector, optimized = measure(lambda: window_features_vectorized(matrix, fs, include_pitch=False), repeats)
    for key in reference:
        if not np.allclose(reference[key], optimized[key], rtol=1e-5, atol=1e-4, equal_nan=True):
            raise AssertionError(f"Reference and optimized feature {key} disagree")
    t_old, old_summary = measure(lambda: summarize_reference(reference, actors), repeats)
    t_new, new_summary = measure(lambda: summarize_optimized(optimized, actors), repeats)
    pd.testing.assert_frame_equal(old_summary, new_summary, check_exact=False, rtol=1e-5, atol=1e-4)
    rows = [
        ("WAV ingestion", t_disk, t_mem),
        ("Four acoustic features (excluding identical pYIN)", t_loop, t_vector),
        ("Feature summary table", t_old, t_new),
    ]
    return pd.DataFrame([{"Stage": label, "Reference [s]": old,
                          "Optimized [s]": new, "Speedup (reference / optimized)": old / new}
                         for label, old, new in rows])


def _sampling_rate(zip_path: str | Path, member: str) -> int:
    """Read sample rate for benchmark comparability."""
    with zipfile.ZipFile(zip_path) as archive:
        return read_wav_in_memory(archive, member)[1]


def main() -> None:
    """Write the benchmark table and provenance metadata to the results folder."""
    parser = argparse.ArgumentParser(description="Compare reference and optimized algorithm timings")
    parser.add_argument("--zip", default=str(ROOT / "data" / "raw" / "synthetic_demo.zip"))
    parser.add_argument("--actors", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    stats = benchmark(args.zip, args.actors, args.repeats)
    results = ROOT / "results" / "tables"
    results.mkdir(parents=True, exist_ok=True)
    stats.to_csv(results / "benchmark.csv", index=False)
    plot_benchmark_comparison(stats, ROOT / "results" / "figures")
    (results / "benchmark_metadata.json").write_text(json.dumps({
        "dataset": "synthetic demo" if Path(args.zip).resolve() == (ROOT / "data/raw/synthetic_demo.zip").resolve() else "user-supplied zip",
        "actors": args.actors, "repetitions": args.repeats,
        "strategy": "minimum elapsed time (seconds); stages measured independently; pYIN excluded from feature timing",
    }, indent=2))
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
