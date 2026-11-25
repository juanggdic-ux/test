"""Executable Lab 3 analysis pipeline; demo data works without a network connection.

Examples:
    python run_pipeline.py
    python run_pipeline.py --zip data/raw/Audio_Speech_Actors_01-24.zip --actors 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.audio import load_paired_audios, truncate_matched_groups
from src.analysis.features import window_features_vectorized
from src.analysis.statistics import summarize_optimized, paired_tests
from src.analysis.classification import grouped_classification, DEFAULT_COLUMNS
from src.visualization.plots import plot_feature_comparison, plot_roc

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "data" / "raw" / "synthetic_demo.zip"


def run(zip_path: str | Path = DEMO, n_actors: int = 6, output_dir: str | Path = ROOT / "results") -> dict:
    """Run ingest, truncate, features, paired tests, grouped CV, and saved figures."""
    output = Path(output_dir)
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    actors, recordings, fs = load_paired_audios(zip_path, n_actors)
    matrices = truncate_matched_groups(recordings)
    summaries = {}
    for condition, samples in matrices.items():
        features = window_features_vectorized(samples, fs)
        summaries[condition] = summarize_optimized(features, actors)
        summaries[condition].to_csv(tables / f"{condition.lower()}_features.csv")
    tests = paired_tests(summaries["Neutral"], summaries["Sad"])
    tests.to_csv(tables / "paired_statistics.csv")
    scores, predictions = grouped_classification(summaries["Neutral"], summaries["Sad"])
    scores.to_csv(tables / "classification.csv")
    predictions.to_csv(tables / "cross_validated_predictions.csv", index=False)
    selected = list(DEFAULT_COLUMNS)
    paths = plot_feature_comparison(summaries["Neutral"], summaries["Sad"], figures, selected)
    paths.append(plot_roc(predictions, figures))
    return {"actors": actors, "sampling_rate": fs, "summaries": summaries,
            "tests": tests, "scores": scores, "figures": paths}


def main() -> None:
    """Parse command-line options and execute the requested pipeline."""
    parser = argparse.ArgumentParser(description="Lab 3: reproducible paired emotional speech analysis")
    parser.add_argument("--zip", default=str(DEMO), help="RAVDESS speech ZIP path (default: bundled synthetic demo)")
    parser.add_argument("--actors", type=int, default=6, help="Maximum matched actors to analyse (up to 24 for RAVDESS)")
    parser.add_argument("--output", default=str(ROOT / "results"), help="Where to write figures and tables")
    arguments = parser.parse_args()
    result = run(arguments.zip, arguments.actors, arguments.output)
    print(f"Processed {len(result['actors'])} matched actors at {result['sampling_rate']} Hz")
    print(result["scores"].round(3).to_string())
    print("Results:", Path(arguments.output).resolve())
    if Path(arguments.zip).resolve() == DEMO.resolve():
        print("DEMO DATA: artificial waveforms only; do not interpret scores as RAVDESS findings.")


if __name__ == "__main__":
    main()
