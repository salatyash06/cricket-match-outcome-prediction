# Cricket Match Outcome Prediction — design

Build a beginner-friendly local Python project, runnable in VS Code, with CSV cleaning, historical features, Logistic Regression / Random Forest / XGBoost comparison, saved best pipeline, and a Streamlit UI.

Predictions happen after the toss and before play. Recent form and runs/wickets refer only to a team's earlier matches, never the match being predicted. Dates are day-level; matches on the same day cannot inform one another. This project demonstrates T20 cricket, using defaults of 0.5 win rate, 150 runs and 6 wickets when there is no history. Sample data is reproducible and synthetic, never presented as actual historical results.

Use a small `cricket_predictor` package: `data.py` cleans CSV rows, `features.py` builds historical statistics, `modeling.py` trains and evaluates pipelines, `predict.py` loads the saved artifact and predicts. `train.py` is the training CLI; `app.py` is the Streamlit app. `scripts/generate_sample_data.py` produces a bundled demo CSV. Tests cover leakage, invalid rows, chronology, model persistence and UI interaction.

CSV fields: match_id,date,team1,team2,venue,toss_winner,toss_decision,winner,team1_runs,team1_wickets,team2_runs,team2_wickets. Optional data_source identifies synthetic data. Dates use YYYY-MM-DD. Scores belong to the named team. Unusable outcomes (tie, abandoned, no result) are excluded with counts; invalid numeric scores become missing values. Impossible teams/toss/date rows are removed. Conflicting duplicate IDs raise an actionable error.

Binary target: 1 means team1 wins, 0 means team2 wins. Categorical fields are one-hot encoded with unknown categories supported. Numeric features are imputed and standardized within fitted scikit-learn pipelines. Chronological day groups are split approximately 60/20/20. Select by validation accuracy, then lower log loss. Refit each candidate on train+validation for test comparison without changing the selected model. Refit the selected architecture on all available data for the saved future-prediction pipeline. Report this distinction clearly. Test evaluation uses earlier completed test matches as rolling history, consistent with sequential forecasting.

The UI offers two distinct teams, venue, toss winner, toss decision and a date after stored history, shows computed recent statistics, predicted winner and both probabilities, plus comparison and confusion matrix. Unknown teams are supported by the Python API using history defaults and shown as warnings. Probabilities are model estimates, not guarantees or calibrated betting odds.

Python 3.10–3.12, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Streamlit, joblib and pytest. No network services are needed at runtime. Include requirements, Windows/macOS/Linux setup commands, CSV schema, generated artifacts, and a ZIP for easy transfer.
