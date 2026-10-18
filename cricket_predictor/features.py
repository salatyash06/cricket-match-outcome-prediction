"""After-toss features using only matches completed on strictly earlier days."""

from collections import defaultdict, deque
from numbers import Integral

import numpy as np
import pandas as pd

from .data import clean_matches, normalize_match_date


CATEGORICAL_FEATURES = ["team1", "team2", "venue", "toss_winner", "toss_decision"]
NUMERIC_FEATURES = [
    "team1_recent_form", "team2_recent_form", "team1_avg_runs", "team2_avg_runs",
    "team1_avg_wickets", "team2_avg_wickets",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _validate_window(window):
    if isinstance(window, bool) or not isinstance(window, Integral) or window <= 0:
        raise ValueError("window must be a positive integer.")
    return int(window)


def _team_stats(records):
    if not records:
        return .5, 150.0, 6.0
    wins, runs, wickets = zip(*records)
    known_runs = [value for value in runs if pd.notna(value)]
    known_wickets = [value for value in wickets if pd.notna(value)]
    return (
        float(np.mean(wins)),
        float(np.mean(known_runs)) if known_runs else 150.0,
        float(np.mean(known_wickets)) if known_wickets else 6.0,
    )


def _feature_row(team1, team2, venue, toss_winner, toss_decision, histories):
    form1, runs1, wickets1 = _team_stats(histories.get(team1, ()))
    form2, runs2, wickets2 = _team_stats(histories.get(team2, ()))
    return {
        "team1": team1, "team2": team2, "venue": venue,
        "toss_winner": toss_winner, "toss_decision": toss_decision,
        "team1_recent_form": form1, "team2_recent_form": form2,
        "team1_avg_runs": runs1, "team2_avg_runs": runs2,
        "team1_avg_wickets": wickets1, "team2_avg_wickets": wickets2,
    }


def _remember_match(row, histories):
    histories[row.team1].append((int(row.winner == row.team1), row.team1_runs, row.team1_wickets))
    histories[row.team2].append((int(row.winner == row.team2), row.team2_runs, row.team2_wickets))


def build_training_features(matches, window=5):
    """Return X and y in cleaned chronological order; 1 means team1 won.

    Each day's rows are featurized together before any result from that day is
    remembered. This also prevents match-ID order from leaking same-day results.
    Scores are averages over available values within the last ``window`` matches,
    rather than searching further back to replace missing scores.
    """
    window = _validate_window(window)
    ordered, _ = clean_matches(matches)
    histories = defaultdict(lambda: deque(maxlen=window))
    rows = []
    for _, day_matches in ordered.groupby("date", sort=False):
        for row in day_matches.itertuples(index=False):
            rows.append(_feature_row(row.team1, row.team2, row.venue,
                                     row.toss_winner, row.toss_decision, histories))
        for row in day_matches.itertuples(index=False):
            _remember_match(row, histories)
    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = ordered["winner"].eq(ordered["team1"]).astype(int).rename("team1_wins")
    return X, y


def _nonempty_name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty team name.")
    return value.strip()


def build_match_features(matches, team1, team2, venue, toss_winner, toss_decision,
                         match_date, window=5):
    """Return one prediction row, ignoring history on/after ``match_date``.

    New teams receive form/runs/wickets defaults of 0.5/150/6. Missing venues
    become ``Unknown``. Dates, datetime objects and pandas timestamps are accepted
    consistently at day resolution.
    """
    window = _validate_window(window)
    team1 = _nonempty_name(team1, "team1")
    team2 = _nonempty_name(team2, "team2")
    if team1 == team2:
        raise ValueError("team1 and team2 must be different teams.")
    toss_winner = _nonempty_name(toss_winner, "toss_winner")
    if toss_winner not in (team1, team2):
        raise ValueError("toss_winner must be one of the two participating teams.")
    if not isinstance(toss_decision, str) or toss_decision.strip().lower() not in ("bat", "field"):
        raise ValueError("toss_decision must be 'bat' or 'field'.")
    toss_decision = toss_decision.strip().lower()
    prediction_day = normalize_match_date(match_date)
    if pd.isna(prediction_day):
        raise ValueError("match_date must be a valid date, preferably YYYY-MM-DD.")
    venue = venue.strip() if isinstance(venue, str) and venue.strip() else "Unknown"

    ordered, _ = clean_matches(matches)
    prior_matches = ordered.loc[ordered["date"] < prediction_day]
    histories = defaultdict(lambda: deque(maxlen=window))
    for row in prior_matches.itertuples(index=False):
        _remember_match(row, histories)
    row = _feature_row(team1, team2, venue, toss_winner, toss_decision, histories)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
