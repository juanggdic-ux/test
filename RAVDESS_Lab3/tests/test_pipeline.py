"""Automated tests for filename parsing, shape, matching, summary and CV."""

from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocessing.audio import parse_filename, choose_matched_members, load_paired_audios, truncate_matched_groups
from src.analysis.features import window_features_vectorized, window_features_reference
from src.analysis.statistics import summarize_optimized, summarize_reference, paired_tests
from src.analysis.classification import grouped_classification

DEMO = Path(__file__).resolve().parents[1] / 'data/raw/synthetic_demo.zip'


def test_ravdess_codes_and_matching():
    """Use emotion codes, not arbitrary offsets within archive names."""
    name = 'Actor_01/03-01-04-01-01-01-01.wav'
    assert parse_filename(name)['emotion'] == '04'
    actors, paths = choose_matched_members([name, 'Actor_01/03-01-01-01-01-01-01.wav'], n_actors=1)
    assert actors == ['01']
    assert len(paths['Neutral']) == len(paths['Sad']) == 1


def test_features_and_summary_equivalent():
    """Vectorized features and column MultiIndex agree with the baseline."""
    ids, data, fs = load_paired_audios(DEMO, n_actors=3)
    matrix = truncate_matched_groups(data)['Neutral']
    old = window_features_reference(matrix, fs, include_pitch=False)
    new = window_features_vectorized(matrix, fs, include_pitch=False)
    for name in old:
        assert np.allclose(old[name], new[name], rtol=1e-5, atol=1e-4)
    a = summarize_reference(old, ids)
    b = summarize_optimized(new, ids)
    pd.testing.assert_frame_equal(a, b, rtol=1e-5, atol=1e-4)
    assert b.index.name == 'Actor'
    assert b.columns.names == ['Characteristic', 'Statistic']


def test_pairwise_statistics_and_actor_cv():
    """Matched 3-actor demo supports paired hypothesis tests and grouped CV."""
    ids, data, fs = load_paired_audios(DEMO, n_actors=3)
    matrices = truncate_matched_groups(data)
    tables = {}
    for condition, matrix in matrices.items():
        tables[condition] = summarize_optimized(window_features_vectorized(matrix, fs), ids)
    tests = paired_tests(tables['Neutral'], tables['Sad'])
    assert len(tests) == 5 * 6
    assert 'Holm-adjusted p' in tests.columns
    scores, prediction = grouped_classification(tables['Neutral'], tables['Sad'])
    assert len(prediction) == 3 * 2 * 2  # actors x conditions x models
    assert scores['ROC-AUC'].between(0, 1).all()
