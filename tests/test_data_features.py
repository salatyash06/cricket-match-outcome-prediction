"""Behavioral checks for cleaning and strictly historical match features."""

from importlib import import_module
import json

import numpy as np
import pandas as pd
import pytest


def api(module):
    """Keep the first red run explicit when a component does not exist yet."""
    try:
        return import_module(f"cricket_predictor.{module}")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Required {module} component is not implemented: {exc}")


def match(match_id="m1", date="2024-01-01", **changes):
    row = dict(
        match_id=match_id, date=date, team1="India", team2="Australia",
        venue="Melbourne", toss_winner="India", toss_decision="bat",
        winner="India", team1_runs=170, team1_wickets=5,
        team2_runs=150, team2_wickets=8,
    )
    row.update(changes)
    return row


def clean(rows):
    return api("data").clean_matches(pd.DataFrame(rows))[0]


def test_cleaning_normalizes_categories_dates_and_order():
    data = api("data")
    frame = pd.DataFrame([
        match(" b ", "2024-01-02 18:30:00", team1=" India ",
              toss_winner=" India ", winner=" India ", toss_decision=" BAT ", venue=None),
        match("a", "2024-01-01"),
    ])
    result, report = data.clean_matches(frame)
    assert result.match_id.tolist() == ["a", "b"]
    assert result.date.tolist() == list(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    assert result.iloc[1].venue == "Unknown"
    assert result.iloc[1].team1 == "India"
    assert result.iloc[1].toss_decision == "bat"
    assert report["input_rows"] == report["output_rows"] == 2
    json.dumps(report)
    assert frame.iloc[0].team1 == " India "  # Caller data is not mutated.


def test_invalid_rows_are_removed_with_exclusive_counts():
    rows = [match("good"), match("date", date="not a date"),
            match("teams", team2="India"), match("toss", toss_winner="England"),
            match("decision", toss_decision="bowl"), match("tie", winner="tie"),
            match("empty", team1=None), match(None)]
    result, report = api("data").clean_matches(pd.DataFrame(rows))
    assert result.match_id.tolist() == ["good"]
    assert report["dropped_rows"] == 7
    assert sum(report["dropped_by_reason"].values()) == 7


def test_invalid_scores_become_missing_and_missing_columns_are_added():
    rows = [match(team1_runs="bad", team2_runs=501, team1_wickets=-1, team2_wickets=11)]
    result, report = api("data").clean_matches(pd.DataFrame(rows))
    scores = ["team1_runs", "team2_runs", "team1_wickets", "team2_wickets"]
    assert result[scores].isna().all().all()
    assert sum(report["invalid_score_values"].values()) == 4
    frame = pd.DataFrame([match()]).drop(columns=scores)
    result, report = api("data").clean_matches(frame)
    assert result[scores].isna().all().all()
    assert set(report["missing_score_columns"]) == set(scores)


def test_fractional_scores_are_missing_but_integral_decimal_scores_are_valid():
    rows = [match("fractional", team1_runs="170.5", team1_wickets="3.2",
                  team2_runs="149.9", team2_wickets="7.5"),
            match("integral", team1_runs="170.0", team1_wickets="3.0",
                  team2_runs="0.0", team2_wickets="10.0")]
    result, report = api("data").clean_matches(pd.DataFrame(rows))
    scores = api("data").SCORE_COLUMNS
    assert result.loc[result.match_id.eq("fractional"), scores].isna().all().all()
    assert result.loc[result.match_id.eq("integral"), scores].iloc[0].tolist() == [170, 3, 0, 10]
    assert report["invalid_score_values"] == dict.fromkeys(scores, 1)


@pytest.mark.parametrize("date_value", ["today", "now", "2024", "01/02/2024"])
def test_cleaning_rejects_relative_or_ambiguous_dates(date_value):
    result, report = api("data").clean_matches(pd.DataFrame([match(date=date_value)]))
    assert result.empty
    assert report["dropped_by_reason"]["invalid_date"] == 1


def test_required_columns_have_actionable_error():
    with pytest.raises(ValueError, match="Missing required.*winner"):
        api("data").clean_matches(pd.DataFrame([match()]).drop(columns=["winner"]))


def test_exact_duplicate_ids_are_removed_and_conflicts_raise():
    result, report = api("data").clean_matches(pd.DataFrame([match(), match()]))
    assert len(result) == 1
    assert report["duplicates_removed"] == 1
    with pytest.raises(ValueError, match="[Cc]onflicting.*m1"):
        api("data").clean_matches(pd.DataFrame([match(), match(winner="Australia")]))


def test_csv_loader_preserves_text_match_ids(tmp_path):
    path = tmp_path / "matches.csv"
    pd.DataFrame([match("0001")]).to_csv(path, index=False)
    result, report = api("data").load_matches(path)
    assert result.match_id.tolist() == ["0001"]
    assert report["output_rows"] == 1


@pytest.mark.parametrize("duplicate_header", ["winner", " winner "])
def test_csv_loader_rejects_duplicate_headers_before_pandas_renames_them(tmp_path, duplicate_header):
    path = tmp_path / "duplicate_headers.csv"
    frame = pd.DataFrame([match()])
    frame[duplicate_header + "_copy"] = "Australia"
    frame.columns = [*frame.columns[:-1], duplicate_header]
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Duplicate CSV column names"):
        api("data").load_matches(path)


def test_first_match_uses_defaults_and_target_maps_to_team1():
    features = api("features")
    X, y = features.build_training_features(clean([match()]))
    assert X.columns.tolist() == features.FEATURE_COLUMNS
    assert X.iloc[0][features.NUMERIC_FEATURES].tolist() == [.5, .5, 150, 150, 6, 6]
    assert y.tolist() == [1]


def test_history_uses_both_slots_and_independent_score_missingness():
    history = clean([
        match("m1", "2024-01-01"),
        match("m2", "2024-01-02", team1="Australia", team2="India",
              winner="Australia", team1_runs=180, team1_wickets=4,
              team2_runs=np.nan, team2_wickets=7),
    ])
    X = api("features").build_match_features(history, "India", "Australia", "Melbourne",
                                             "Australia", "field", "2024-01-03")
    row = X.iloc[0]
    assert row.team1_recent_form == row.team2_recent_form == .5
    assert row.team1_avg_runs == 170
    assert row.team2_avg_runs == 165
    assert row.team1_avg_wickets == row.team2_avg_wickets == 6


def test_current_and_future_outcomes_and_scores_cannot_change_features():
    features = api("features")
    history = clean([match(f"m{i}", f"2024-01-0{i}") for i in range(1, 5)])
    original, y_original = features.build_training_features(history)
    changed = history.copy()
    changed.loc[2:, "winner"] = "Australia"
    changed.loc[2:, "team1_runs"] = 300
    changed.loc[2:, "team1_wickets"] = 1
    altered, y_altered = features.build_training_features(changed)
    pd.testing.assert_frame_equal(original.iloc[:3], altered.iloc[:3])
    assert y_original.iloc[2] != y_altered.iloc[2]
    assert original.iloc[3].team1_recent_form != altered.iloc[3].team1_recent_form


def test_same_day_matches_cannot_inform_each_other():
    history = clean([match("m1", "2024-01-01"), match("m2", "2024-01-01 18:00:00")])
    features = api("features")
    X, _ = features.build_training_features(history)
    assert X.team1_recent_form.tolist() == [.5, .5]
    assert X.team1_avg_runs.tolist() == [150, 150]
    prediction = features.build_match_features(history, "India", "Australia", "Melbourne",
                                               "India", "bat", pd.Timestamp("2024-01-01 23:00"))
    assert prediction.iloc[0].team1_avg_runs == 150


def test_rolling_window_uses_latest_matches_in_either_slot():
    history = clean([match(f"m{i}", f"2024-01-0{i}",
                           winner="Australia" if i == 1 else "India", team1_runs=100 + 10*i)
                     for i in range(1, 7)])
    X = api("features").build_match_features(history, "India", "Australia", "Melbourne",
                                             "India", "bat", "2024-01-07", window=5)
    assert X.iloc[0].team1_recent_form == 1
    assert X.iloc[0].team1_avg_runs == 140


def test_unknown_team_gets_history_defaults():
    X = api("features").build_match_features(clean([match()]), "New Team", "Australia",
                                             "New Venue", "New Team", "field", "2024-01-02")
    assert X.iloc[0].team1_recent_form == .5
    assert X.iloc[0].team1_avg_runs == 150
    assert X.iloc[0].team1_avg_wickets == 6
    assert X.iloc[0].team2_recent_form == 0


@pytest.mark.parametrize("changes", [
    {"team2": "India"}, {"team1": ""}, {"toss_winner": "England"},
    {"toss_decision": "bowl"}, {"match_date": "bad"}, {"window": 0},
    {"window": 1.5}, {"window": True},
])
def test_prediction_rejects_invalid_requests(changes):
    kwargs = dict(team1="India", team2="Australia", venue="Melbourne",
                  toss_winner="India", toss_decision="bat", match_date="2024-01-02")
    kwargs.update(changes)
    with pytest.raises(ValueError):
        api("features").build_match_features(clean([match()]), **kwargs)


@pytest.mark.parametrize("date_value", ["2024-01-02", " 2024-01-02 ",
                                         "2024-01-02T18:00:00+02:00",
                                         pd.Timestamp("2024-01-02"),
                                         pd.Timestamp("2024-01-02").date(),
                                         pd.Timestamp("2024-01-02").to_pydatetime()])
def test_prediction_accepts_common_date_types(date_value):
    X = api("features").build_match_features(clean([match()]), "India", "Australia", "Melbourne",
                                             "India", "bat", date_value)
    assert X.iloc[0].team1_recent_form == 1
