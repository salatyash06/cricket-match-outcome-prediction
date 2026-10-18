"""Train comparable pipelines, evaluate future dates, and save the best model."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Also permits chart export on computers without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import load_matches
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_training_features


def chronological_split(matches):
    """Return positional train/validation/test indices without dividing a day.

    Splits use approximately 60/20/20 percent of distinct dates. With multiple
    matches per date, row percentages can differ. At least 30 completed matches
    across five distinct dates are required for this educational workflow.
    """
    if len(matches) < 30:
        raise ValueError("Training needs at least 30 usable completed matches.")
    dates = pd.to_datetime(matches["date"]).dt.normalize()
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < 5:
        raise ValueError("Training needs at least 5 distinct dates for chronological splits.")
    train_end = max(1, int(len(unique_dates) * 0.6))
    validation_end = min(len(unique_dates) - 1, max(train_end + 1, int(len(unique_dates) * 0.8)))
    return (
        np.flatnonzero(dates < unique_dates[train_end]),
        np.flatnonzero((dates >= unique_dates[train_end]) & (dates < unique_dates[validation_end])),
        np.flatnonzero(dates >= unique_dates[validation_end]),
    )


def make_pipeline(classifier):
    """Fit encoding and scaling with the classifier to prevent preprocessing leaks."""
    categories = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numbers = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("categories", categories, CATEGORICAL_FEATURES),
        ("numbers", numbers, NUMERIC_FEATURES),
    ])
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def candidate_models():
    """Small, reproducible CPU models that train quickly on a laptop."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=7, min_samples_leaf=4, random_state=42, n_jobs=1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=1,
        ),
    }


def evaluate_model(model, X, y):
    """Return JSON-friendly scores and predicted binary labels."""
    predicted = model.predict(X)
    probabilities = model.predict_proba(X)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
    }, predicted


def save_charts(metrics, output_dir):
    """Write portable Matplotlib PNGs and close figures to release memory."""
    comparison = pd.DataFrame(metrics["candidates"])
    fig, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(comparison))
    axis.bar(positions - .18, comparison.validation_accuracy, width=.36,
             label="Validation", color="#8fc9aa")
    axis.bar(positions + .18, comparison.test_accuracy, width=.36,
             label="Test", color="#177349")
    axis.set_xticks(positions, comparison["name"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Cricket match outcome: model comparison")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        np.asarray(metrics["test_confusion_matrix"]),
        display_labels=["Team 2 wins", "Team 1 wins"],
    ).plot(ax=axis, cmap="Greens", colorbar=False, values_format="d")
    axis.set_title(f"{metrics['selected_model']} — held-out test")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def train_models(data_path, output_dir, window=5):
    """Train all three candidates and return the saved inference artifact.

    Selection uses validation only. Each candidate is then refitted on train +
    validation before test evaluation. Finally, the selected architecture is
    refitted on all history for future inference; the test score is a score of
    the earlier evaluation model, not a second evaluation of that final refit.
    """
    matches, cleaning_report = load_matches(data_path)
    train, validation, test = chronological_split(matches)
    X, y = build_training_features(matches, window=window)
    if y.iloc[train].nunique() != 2:
        raise ValueError(
            "The training period must contain both team1 and team2 wins. "
            "Check winner labels and add more earlier matches."
        )
    templates = {name: make_pipeline(model) for name, model in candidate_models().items()}
    candidates = []
    for name, template in templates.items():
        model = clone(template).fit(X.iloc[train], y.iloc[train])
        scores, _ = evaluate_model(model, X.iloc[validation], y.iloc[validation])
        candidates.append({"name": name, "validation_accuracy": scores["accuracy"],
                           "validation_log_loss": scores["log_loss"]})

    # Do not change this choice after inspecting test performance.
    selected = min(candidates, key=lambda row: (
        -row["validation_accuracy"], row["validation_log_loss"], row["name"]
    ))["name"]
    train_validation = np.concatenate([train, validation])
    test_predictions = {}
    for row in candidates:
        model = clone(templates[row["name"]]).fit(X.iloc[train_validation], y.iloc[train_validation])
        scores, predicted = evaluate_model(model, X.iloc[test], y.iloc[test])
        row.update(test_accuracy=scores["accuracy"], test_log_loss=scores["log_loss"])
        test_predictions[row["name"]] = predicted

    majority_class = int(y.iloc[train].mode().iloc[0])
    selected_scores = next(row for row in candidates if row["name"] == selected)
    split = {}
    for name, indices in [("train", train), ("validation", validation), ("test", test)]:
        dates = matches.iloc[indices]["date"]
        split[name] = {"rows": len(indices), "start": dates.min().date().isoformat(),
                       "end": dates.max().date().isoformat()}
    metrics = {
        "selected_model": selected,
        "selection_metric": "validation_accuracy",
        "candidates": candidates,
        "test_accuracy": selected_scores["test_accuracy"],
        "test_confusion_matrix": confusion_matrix(
            y.iloc[test], test_predictions[selected], labels=[0, 1]
        ).tolist(),
        "baseline_test_accuracy": float(accuracy_score(y.iloc[test], np.full(len(test), majority_class))),
        "baseline_class": majority_class,
        "split": split,
        "cleaning_report": cleaning_report,
        "evaluation_note": "Test models fit train+validation; saved model refit on all history. "
                           "Features use strictly earlier dates, including earlier test results.",
    }
    sources = matches.get("data_source", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
    metadata = {
        "schema_version": 1,
        "model_name": selected,
        "history_start": matches.date.min().date().isoformat(),
        "history_end": matches.date.max().date().isoformat(),
        "teams": sorted(set(matches.team1) | set(matches.team2)),
        "venues": sorted(matches.venue.unique().tolist()),
        "feature_window": int(window),
        "feature_columns": list(X.columns),
        "synthetic_data": bool(sources.str.contains("synthetic", regex=False).any()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "library_versions": {name: version(name) for name in
                             ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "joblib"]},
    }
    final_model = clone(templates[selected]).fit(X, y)
    artifact = {"model": final_model, "metadata": metadata, "history": matches, "metrics": metrics}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(candidates).to_csv(output_dir / "model_comparison.csv", index=False)
    save_charts(metrics, output_dir)
    # Replace only after a complete write, so the app never loads half a pickle.
    temporary_model = output_dir / "best_model.joblib.tmp"
    joblib.dump(artifact, temporary_model, compress=3)
    temporary_model.replace(output_dir / "best_model.joblib")
    return artifact
