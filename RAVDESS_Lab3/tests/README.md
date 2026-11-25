# Tests

Run `python -m pytest -q` from the project root.

Tests cover metadata-aware emotion matching, equivalence of loop-based and vectorized feature calculations, agreement between both statistical DataFrame builders, the required two-level column index, paired significance summaries, and leave-one-actor-out classification.

The real RAVDESS archive is not required to test the pipeline; tests use the bundled deterministic synthetic ZIP.
