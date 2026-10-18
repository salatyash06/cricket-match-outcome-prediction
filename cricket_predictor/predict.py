"""Load a trusted model artifact and predict a future match after the toss."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import normalize_match_date
from .features import build_match_features


def load_artifact(path):
    """Load only joblib files you created or trust: pickle can execute code."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No saved model at {path}. Run python train.py first.")
    artifact = joblib.load(path)
    required = {"model", "metadata", "history", "metrics"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("This is not a cricket prediction artifact. Run python train.py again.")
    if artifact["metadata"].get("schema_version") != 1:
        raise ValueError("Unsupported model artifact version. Run python train.py again.")
    return artifact


def predict_match(artifact, team1, team2, venue, toss_winner, toss_decision, match_date):
    """Return the winner, both probabilities, input features and any warnings.

    The model is trained on all stored matches. An earlier-date prediction
    would therefore leak training outcomes even if historical features were
    filtered correctly, so only dates after that history are accepted here.
    """
    metadata = artifact["metadata"]
    prediction_day = normalize_match_date(match_date)
    if pd.isna(prediction_day):
        raise ValueError("match_date must be a valid date, preferably YYYY-MM-DD.")
    if prediction_day <= pd.Timestamp(metadata["history_end"]):
        raise ValueError(f"Choose a match date after {metadata['history_end']}, the end of saved training history.")
    features = build_match_features(
        artifact["history"], team1, team2, venue, toss_winner, toss_decision,
        prediction_day, window=metadata["feature_window"],
    )
    first, second = features.iloc[0][["team1", "team2"]]
    model = artifact["model"]
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    if set(classes) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("Saved model has invalid outcome probabilities. Retrain with python train.py.")
    probability_first = float(probabilities[classes.index(1)])
    # The complement avoids tiny floating-point sum errors for float32 outputs.
    by_team = {first: probability_first, second: 1.0 - probability_first}
    winner = first if probability_first >= 0.5 else second
    warnings = []
    for team in [first, second]:
        if team not in metadata["teams"]:
            warnings.append(f"{team} was not in the training data; default history statistics are used.")
    if features.iloc[0]["venue"] not in metadata["venues"]:
        warnings.append("This venue was not in the training data; its category has no learned effect.")
    return {"winner": winner, "win_probability": by_team[winner],
            "probabilities": by_team, "features": features, "warnings": warnings}
