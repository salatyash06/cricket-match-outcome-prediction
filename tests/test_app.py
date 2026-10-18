"""Exercise the local app as a user, using an isolated trained demo model."""

from pathlib import Path
import shutil

import pytest
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained_app_path(tmp_path_factory):
    """Keep UI tests independent of artifacts generated in the user's project."""
    from cricket_predictor.modeling import train_models

    directory = tmp_path_factory.mktemp("trained_app")
    app_path = directory / "app.py"
    shutil.copyfile(PROJECT_ROOT / "app.py", app_path)
    (directory / "data").mkdir()
    sample_path = directory / "data" / "sample_matches.csv"
    shutil.copyfile(PROJECT_ROOT / "data" / "sample_matches.csv", sample_path)
    train_models(sample_path, directory / "artifacts")
    return app_path


@pytest.fixture
def app(trained_app_path):
    return AppTest.from_file(str(trained_app_path), default_timeout=30).run()


def predict(app):
    next(button for button in app.button if button.label == "Predict match").click().run()
    assert not app.exception
    return app


def probability_metrics(app):
    return [metric for metric in app.metric if metric.label.endswith("win probability")]


def test_app_loads_saved_model_and_comparison(app):
    assert not app.exception
    assert app.title[0].value == "🏏 Cricket Match Predictor"
    assert [tab.label for tab in app.tabs] == ["Predict", "Model performance", "Data guide"]
    assert any("Synthetic demo data" in warning.value for warning in app.warning)
    comparison = next(frame.value for frame in app.dataframe if "Model" in frame.value.columns)
    assert set(comparison["Model"]) == {"Logistic Regression", "Random Forest", "XGBoost"}
    selected = next(metric.value for metric in app.metric if metric.label == "Selected model")
    assert selected in set(comparison["Model"])
    assert not app.success


def test_predict_shows_winner_probabilities_and_recent_history(app):
    team1 = app.selectbox(key="team1").value
    team2 = app.selectbox(key="team2").value
    predict(app)

    assert app.success[0].value in {f"Predicted winner: {team1}", f"Predicted winner: {team2}"}
    metrics = probability_metrics(app)
    assert {metric.label for metric in metrics} == {
        f"{team1} · win probability", f"{team2} · win probability"
    }
    percentages = [float(metric.value.removesuffix("%")) for metric in metrics]
    assert all(0 <= value <= 100 for value in percentages)
    assert sum(percentages) == pytest.approx(100, abs=0.1)
    history = next(frame.value for frame in app.dataframe if "Team" in frame.value.columns)
    assert history["Team"].tolist() == [team1, team2]
    assert history.columns.tolist() == [
        "Team", "Recent win rate", "Average runs scored", "Average wickets lost"
    ]
    assert not history.isna().any().any()


def test_changing_teams_hides_stale_results_and_keeps_teams_distinct(app):
    predict(app)
    previous_team2 = app.selectbox(key="team2").value
    app.selectbox(key="team1").select(previous_team2).run()

    assert not app.exception
    team1 = app.selectbox(key="team1").value
    team2 = app.selectbox(key="team2").value
    assert team1 == previous_team2
    assert team1 != team2
    assert team1 not in app.selectbox(key="team2").options
    assert set(app.selectbox(key="toss_winner").options) == {team1, team2}
    assert not app.success
    assert not probability_metrics(app)

    predict(app)
    history = next(frame.value for frame in app.dataframe if "Team" in frame.value.columns)
    assert history["Team"].tolist() == [team1, team2]


def test_missing_artifact_shows_training_command(tmp_path):
    app_path = tmp_path / "app.py"
    shutil.copyfile(PROJECT_ROOT / "app.py", app_path)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    assert any("No trained model found" in warning.value for warning in app.warning)
    assert any(code.value == "python train.py" for code in app.code)
    assert any(button.label == "Train demo models" for button in app.button)
    assert not app.selectbox


def test_unreadable_artifact_shows_retraining_help(tmp_path):
    app_path = tmp_path / "app.py"
    shutil.copyfile(PROJECT_ROOT / "app.py", app_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "best_model.joblib").write_text("This is not a saved model.", encoding="utf-8")
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    assert any("could not be loaded" in error.value for error in app.error)
    assert any(code.value == "python train.py" for code in app.code)
    assert any(button.label == "Train demo models" for button in app.button)
