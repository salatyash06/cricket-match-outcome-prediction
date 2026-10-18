# Cricket Match Outcome Prediction Implementation Plan

**Goal:** Deliver and verify the complete local project requested by the user.

**Architecture:** A CSV-to-features pipeline feeds three comparable binary classifiers. A saved full-data pipeline and clean history support a simple Streamlit interface.

**Spec:** `docs/design.md`

## Global constraints

- Python 3.10–3.12; no current-match score or future result leakage.
- Synthetic sample data must be labeled in CSV, README, UI and metadata.
- Date groups never cross split boundaries; preprocessing is fitted inside each pipeline.
- User authorization covers creating and testing this project in its own new directory.

## Review focus

- Missing/unusable CSV values and conflicting IDs give actionable feedback.
- A changed result cannot affect that match's features or earlier predictions.
- Same-day rows and split boundaries preserve chronology.
- Predictions map class 1 to team1 and sum to one; unknown categories are accepted.
- Fresh setup, saved artifact loading and UI prediction work together.

## Task 1: Data and features

- [x] Add and verify behavioral tests for cleaning, previous-match-only form and same-day exclusion.
- [x] Implement `load_matches(path) -> (DataFrame, dict)` and `clean_matches(frame) -> (DataFrame, dict)` in `cricket_predictor/data.py`.
- [x] Implement `build_training_features(matches, window=5) -> (X, y)` and `build_match_features(matches, team1, team2, venue, toss_winner, toss_decision, match_date, window=5) -> DataFrame` in `features.py`.
- [x] Generate 720 deterministic synthetic T20 matches and a small format example; verify tests.

## Task 2: Training and prediction

- [x] Write tests for date splitting, training all candidates, artifact round-trip and probability validation.
- [x] Implement `train_models(data_path, output_dir, window=5) -> dict` in `modeling.py`, saving `best_model.joblib`, `metrics.json`, `model_comparison.csv`, `model_comparison.png`, `confusion_matrix.png`.
- [x] Implement `load_artifact(path) -> dict` and `predict_match(artifact, team1, team2, venue, toss_winner, toss_decision, match_date) -> dict` in `predict.py`.
- [x] Add CLI, pinned requirements and project config; verify integration tests and train the supplied CSV.

## Task 3: Interface and documentation

- [x] Build `app.py`, using saved metadata/history and prediction API; include graceful untrained state and training button.
- [x] Write README with VS Code setup, OS-specific environment commands, schema, methodology, artifact details and limitations.
- [x] Exercise app loading, team selection, prediction submission and metrics with Streamlit AppTest.

## Task 4: Verify and deliver

- [x] Independently review project for correctness and gaps; fix material issues with regression checks.
- [x] Run complete pytest suite, train CLI, Streamlit startup/health check, and dependency consistency checks.
- [x] Package source, CSV and trained artifacts without virtualenv or cache files; report runnable commands and observed results.
