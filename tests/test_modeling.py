"""Integration checks for chronology, persistence and probability semantics."""

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def matches_csv(tmp_path):
    rng = np.random.default_rng(42)
    teams = ["India", "Australia", "England", "New Zealand"]
    rows = []
    for i in range(120):
        first, second = rng.choice(teams, 2, replace=False)
        rows.append({
            "match_id": f"TEST-{i:03}",
            "date": (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).date(),
            "team1": first, "team2": second, "venue": f"Ground {i % 3}",
            "toss_winner": first if i % 2 else second,
            "toss_decision": "bat" if i % 3 else "field",
            "winner": first if i % 3 else second,
            "team1_runs": 140 + i % 50, "team2_runs": 130 + i % 60,
            "team1_wickets": i % 11, "team2_wickets": (i + 3) % 11,
            "data_source": "synthetic",
        })
    path = tmp_path / "matches.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_split_keeps_dates_together_and_in_time_order():
    from cricket_predictor.modeling import chronological_split

    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20).repeat(3)})
    train, validation, test = chronological_split(frame)
    assert len(train) + len(validation) + len(test) == len(frame)
    assert frame.iloc[train].date.max() < frame.iloc[validation].date.min()
    assert frame.iloc[validation].date.max() < frame.iloc[test].date.min()
    assert len(set(train) & set(test)) == 0


def test_split_rejects_insufficient_history():
    from cricket_predictor.modeling import chronological_split

    with pytest.raises(ValueError, match="30"):
        chronological_split(pd.DataFrame({"date": pd.date_range("2024-01-01", periods=12)}))
    with pytest.raises(ValueError, match="distinct dates"):
        chronological_split(pd.DataFrame({"date": [pd.Timestamp("2024-01-01")] * 30}))


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    # Reuse the CSV fixture's deterministic real-data shape without mocking models.
    from cricket_predictor.modeling import train_models

    directory = tmp_path_factory.mktemp("training")
    path = matches_csv.__wrapped__(directory)
    artifact = train_models(path, directory / "artifacts")
    return artifact, directory / "artifacts"


def test_all_models_are_compared_and_selected_on_validation(trained_artifact):
    artifact, output = trained_artifact
    metrics = artifact["metrics"]
    assert {row["name"] for row in metrics["candidates"]} == {
        "Logistic Regression", "Random Forest", "XGBoost"
    }
    best = sorted(metrics["candidates"], key=lambda row: (
        -row["validation_accuracy"], row["validation_log_loss"], row["name"]
    ))[0]
    assert metrics["selected_model"] == best["name"]
    assert metrics["test_accuracy"] == best["test_accuracy"]
    assert np.array(metrics["test_confusion_matrix"]).sum() == metrics["split"]["test"]["rows"]
    assert artifact["metadata"]["synthetic_data"] is True
    for filename in ["best_model.joblib", "metrics.json", "model_comparison.csv",
                     "model_comparison.png", "confusion_matrix.png"]:
        assert (output / filename).stat().st_size > 0
    assert json.loads((output / "metrics.json").read_text())["selected_model"] == best["name"]


def test_saved_model_predicts_both_probabilities(trained_artifact):
    from cricket_predictor.predict import load_artifact, predict_match

    artifact, output = trained_artifact
    restored = load_artifact(output / "best_model.joblib")
    kwargs = dict(team1="India", team2="Australia", venue="Ground 0",
                  toss_winner="India", toss_decision="bat", match_date="2025-01-01")
    before = predict_match(artifact, **kwargs)
    result = predict_match(restored, **kwargs)
    assert result["probabilities"] == before["probabilities"]
    assert set(result["probabilities"]) == {"India", "Australia"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert all(0 <= p <= 1 for p in result["probabilities"].values())
    probabilities = restored["model"].predict_proba(result["features"])[0]
    class_one = list(restored["model"].classes_).index(1)
    assert result["probabilities"]["India"] == pytest.approx(probabilities[class_one])
    assert result["win_probability"] == max(result["probabilities"].values())
    assert result["winner"] in {"India", "Australia"}


def test_predict_unknown_categories_and_invalid_match(trained_artifact):
    from cricket_predictor.predict import predict_match

    artifact, _ = trained_artifact
    kwargs = dict(team1="New Team", team2="India", venue="New Ground",
                  toss_winner="India", toss_decision="field", match_date="2025-01-01")
    result = predict_match(artifact, **kwargs)
    assert result["warnings"]
    assert result["features"].iloc[0]["team1_recent_form"] == 0.5
    with pytest.raises(ValueError, match="after"):
        predict_match(artifact, **{**kwargs, "match_date": "2024-02-01"})
    with pytest.raises(ValueError, match="after"):
        predict_match(artifact, **{**kwargs, "match_date": artifact["metadata"]["history_end"] + " 23:59:00"})
    with pytest.raises(ValueError):
        predict_match(artifact, **{**kwargs, "team1": "India"})
    with pytest.raises(ValueError):
        predict_match(artifact, **{**kwargs, "toss_winner": "Other"})


def test_single_class_training_data_has_actionable_error(matches_csv, tmp_path):
    from cricket_predictor.modeling import train_models

    frame = pd.read_csv(matches_csv)
    frame["winner"] = frame["team1"]
    frame.to_csv(matches_csv, index=False)
    with pytest.raises(ValueError, match="both"):
        train_models(matches_csv, tmp_path / "output")
