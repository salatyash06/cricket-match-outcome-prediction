"""Read match CSVs and report every discarded row or unusable score."""

import csv
from numbers import Number
from pathlib import Path
import re

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "match_id", "date", "team1", "team2", "venue", "toss_winner",
    "toss_decision", "winner",
]
SCORE_COLUMNS = ["team1_runs", "team1_wickets", "team2_runs", "team2_wickets"]


def normalize_match_date(value):
    """Return a timezone-free, day-level timestamp; invalid input becomes NaT.

    Strings must contain an explicit YYYY-MM-DD date, optionally followed by a
    time. Numeric values are not interpreted as Unix nanoseconds. For timezone-
    aware timestamps the supplied calendar day is retained, since CSV dates are days.
    """
    if isinstance(value, Number) or value is None:
        return pd.NaT
    if isinstance(value, str):
        value = value.strip()
        if not re.match(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}(?:$|[Tt ])", value):
            return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return pd.NaT
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        timestamp = timestamp.normalize()
        # Ensure it can be stored alongside normal datetime64[ns] dates.
        return timestamp.as_unit("ns")
    except (TypeError, ValueError, OverflowError):
        return pd.NaT


def clean_matches(frame):
    """Return cleaned matches in date/ID order and a JSON-serializable report.

    Missing score columns are permitted: their historical statistics fall back
    to documented defaults. Exact duplicate records are retained once; an ID
    naming different records is an error to fix at the source. Drop counts are
    exclusive, so a row with several errors is counted under its first error.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Matches must be a pandas DataFrame.")
    cleaned = frame.copy(deep=True).reset_index(drop=True)
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    if cleaned.columns.duplicated().any():
        raise ValueError("Duplicate CSV column names: give every column a unique name.")
    missing = [column for column in REQUIRED_COLUMNS if column not in cleaned]
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))

    missing_scores = [column for column in SCORE_COLUMNS if column not in cleaned]
    for column in missing_scores:
        cleaned[column] = np.nan
    text_columns = [column for column in REQUIRED_COLUMNS if column != "date"]
    if "data_source" in cleaned:
        text_columns.append("data_source")
    for column in text_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip().replace("", pd.NA)
    cleaned["venue"] = cleaned["venue"].fillna("Unknown")
    cleaned["toss_decision"] = cleaned["toss_decision"].str.lower()

    # Check before coercing scores/dates: two different bad values must not turn
    # into identical NaNs and hide a conflicting match record.
    duplicated_ids = cleaned["match_id"].notna() & cleaned["match_id"].duplicated(keep=False)
    conflicting = []
    for match_id, group in cleaned.loc[duplicated_ids].groupby("match_id", sort=False):
        if len(group.drop_duplicates()) > 1:
            conflicting.append(str(match_id))
    if conflicting:
        examples = ", ".join(conflicting[:5])
        raise ValueError(
            f"Conflicting records share match_id values: {examples}. "
            "Correct the records or assign a distinct ID to each match."
        )
    duplicates = cleaned["match_id"].notna() & cleaned["match_id"].duplicated(keep="first")
    duplicates_removed = int(duplicates.sum())
    cleaned = cleaned.loc[~duplicates].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"].map(normalize_match_date))

    dropped_by_reason = {"duplicate_match_id": duplicates_removed}

    def drop_invalid(mask, reason):
        nonlocal cleaned
        mask = mask.fillna(True)
        dropped_by_reason[reason] = int(mask.sum())
        cleaned = cleaned.loc[~mask].copy()

    drop_invalid(cleaned["match_id"].isna(), "missing_match_id")
    drop_invalid(cleaned["date"].isna(), "invalid_date")
    drop_invalid(cleaned["team1"].isna() | cleaned["team2"].isna()
                 | cleaned["team1"].eq(cleaned["team2"]), "invalid_teams")
    drop_invalid(~(cleaned["toss_winner"].eq(cleaned["team1"])
                   | cleaned["toss_winner"].eq(cleaned["team2"])), "invalid_toss_winner")
    drop_invalid(~cleaned["toss_decision"].isin(["bat", "field"]), "invalid_toss_decision")
    drop_invalid(~(cleaned["winner"].eq(cleaned["team1"])
                   | cleaned["winner"].eq(cleaned["team2"])), "unusable_outcome")

    invalid_scores = {}
    for column in SCORE_COLUMNS:
        raw = cleaned[column]
        values = pd.to_numeric(raw, errors="coerce")
        upper = 10 if column.endswith("wickets") else 500
        valid = values.between(0, upper) & values.mod(1).eq(0)
        present = raw.notna() & raw.astype("string").str.strip().ne("").fillna(False)
        invalid_scores[column] = int((present & ~valid).sum())
        cleaned[column] = values.where(valid, np.nan).astype(float)

    # Object strings are compatible with sklearn's categorical imputers.
    for column in text_columns:
        cleaned[column] = cleaned[column].astype(object).where(cleaned[column].notna(), np.nan)
    cleaned = cleaned.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    report = {
        "input_rows": int(len(frame)),
        "output_rows": int(len(cleaned)),
        "dropped_rows": int(len(frame) - len(cleaned)),
        "duplicates_removed": duplicates_removed,
        "dropped_by_reason": dropped_by_reason,
        "invalid_score_values": invalid_scores,
        "missing_score_columns": missing_scores,
    }
    return cleaned, report


def load_matches(path):
    """Read a CSV without losing leading zeroes in match IDs, then clean it."""
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            # pandas renames duplicate headers (winner -> winner.1), which can
            # otherwise conceal two conflicting outcome columns from cleaning.
            header = next((row for row in csv.reader(handle)
                           if row and (len(row) > 1 or row[0].strip())), [])
            names = [column.strip() for column in header]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate CSV column names: give every column a unique name.")
            handle.seek(0)
            frame = pd.read_csv(handle, dtype=str)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV is empty; include a header and completed match rows.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse the CSV. Check its header and separators: {exc}") from exc
    return clean_matches(frame)
