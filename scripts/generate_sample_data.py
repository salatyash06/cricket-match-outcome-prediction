#!/usr/bin/env python3
"""Generate SYNTHETIC T20 examples; these are not real matches or statistics.

The invented strengths, scoring environments, and home/toss effects below exist
only to give a learning demo a modest, noisy predictive signal. Team and venue
names are recognizable labels, not claims about their actual performance.

Run from the project directory::

    python scripts/generate_sample_data.py --output data/sample_matches.csv --matches 720 --seed 42
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sample_matches.csv"
FIELDS = [
    "match_id", "date", "team1", "team2", "venue", "toss_winner",
    "toss_decision", "winner", "team1_runs", "team1_wickets", "team2_runs",
    "team2_wickets", "data_source",
]

# All numerical parameters are invented for this demonstration.
TEAM_STRENGTH = {
    "India": 0.75,
    "Australia": 0.55,
    "England": 0.45,
    "New Zealand": 0.25,
    "South Africa": 0.10,
    "Pakistan": -0.15,
    "Sri Lanka": -0.45,
    "West Indies": -0.60,
}
# venue: (home team, invented run adjustment, invented chasing advantage)
VENUES = {
    "Wankhede Stadium": ("India", 12, 0.18),
    "Melbourne Cricket Ground": ("Australia", -5, -0.08),
    "Lord's": ("England", -2, 0.04),
    "Eden Park": ("New Zealand", 15, 0.12),
    "Newlands": ("South Africa", -8, 0.05),
    "Gaddafi Stadium": ("Pakistan", 8, 0.10),
    "R. Premadasa Stadium": ("Sri Lanka", -12, -0.10),
    "Kensington Oval": ("West Indies", 0, 0.02),
}


def generate_matches(matches: int = 720, seed: int = 42) -> list[dict]:
    """Return reproducible synthetic records with randomly ordered team slots."""
    if matches < 1:
        raise ValueError("matches must be a positive integer")
    rng = np.random.default_rng(seed)
    teams = list(TEAM_STRENGTH)
    venues = list(VENUES)
    # Every complete cycle includes both slot orderings for every team pair.
    # Shuffling each cycle keeps team1/team2 assignment independent of strength.
    pairings = [(a, b) for a in teams for b in teams if a != b]
    schedule: list[tuple[str, str]] = []
    while len(schedule) < matches:
        schedule.extend(pairings[int(i)] for i in rng.permutation(len(pairings)))

    rows = []
    for index, (team1, team2) in enumerate(schedule[:matches]):
        venue = venues[int(rng.integers(len(venues)))]
        home_team, run_adjustment, chase_advantage = VENUES[venue]
        toss_winner = (team1, team2)[int(rng.integers(2))]
        toss_decision = "field" if rng.random() < 0.58 else "bat"
        other_team = team2 if toss_winner == team1 else team1
        batting_first = toss_winner if toss_decision == "bat" else other_team
        chasing = team2 if batting_first == team1 else team1

        strength_difference = TEAM_STRENGTH[team1] - TEAM_STRENGTH[team2]
        home_effect = 0.20 * ((team1 == home_team) - (team2 == home_team))
        toss_effect = 0.10 if toss_winner == team1 else -0.10
        chase_effect = chase_advantage if chasing == team1 else -chase_advantage
        log_odds = (
            1.6 * strength_difference + home_effect + toss_effect + chase_effect
            + float(rng.normal(0, 0.35))
        )
        team1_probability = 1.0 / (1.0 + np.exp(-log_odds))
        winner = team1 if rng.random() < team1_probability else team2

        first_runs = int(np.clip(round(
            164 + run_adjustment + 12 * TEAM_STRENGTH[batting_first]
            - 5 * TEAM_STRENGTH[chasing] + rng.normal(0, 22)
        ), 95, 245))
        first_wickets = int(np.clip(round(rng.normal(6.0, 1.8)), 1, 10))
        if winner == batting_first:
            margin = int(np.clip(round(rng.gamma(shape=2.0, scale=10.0)), 1, 65))
            chase_runs = first_runs - margin
            chase_wickets = int(np.clip(round(rng.normal(7.5, 1.7)), 2, 10))
        else:
            chase_runs = first_runs + int(rng.integers(1, 7))
            chase_wickets = int(np.clip(round(rng.normal(4.5, 1.8)), 0, 9))
        scores = {
            batting_first: (first_runs, first_wickets),
            chasing: (chase_runs, chase_wickets),
        }
        rows.append({
            "match_id": f"SYN{index + 1:05d}",
            "date": (date(2022, 1, 1) + timedelta(days=index)).isoformat(),
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "winner": winner,
            "team1_runs": scores[team1][0],
            "team1_wickets": scores[team1][1],
            "team2_runs": scores[team2][0],
            "team2_wickets": scores[team2][1],
            "data_source": "synthetic",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--matches", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.matches < 1:
        parser.error("--matches must be a positive integer")
    if args.seed < 0:
        parser.error("--seed must be a nonnegative integer")
    rows = generate_matches(args.matches, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic matches to {args.output}")


if __name__ == "__main__":
    main()
