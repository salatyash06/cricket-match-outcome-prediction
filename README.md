# Cricket Match Outcome Prediction

A complete, local T20 cricket machine-learning project for learning in VS Code. Clean match CSVs, calculate recent team statistics, compare **Logistic Regression, Random Forest and XGBoost**, save the selected model, and make predictions in a Streamlit app.

**The included 720-match dataset is synthetic.** Its teams, scores and outcomes are generated examples, not historical results. Accuracy on it demonstrates the code and evaluation workflow, not real-world forecasting ability. Predicted probabilities are model estimates and are not calibrated probabilities or betting odds.

## What you can do

- Predict a match after the toss and before play using teams, venue and toss details.
- Inspect each team's recent win rate, average runs scored and average wickets lost.
- Compare validation and test accuracy, log loss and the selected model's confusion matrix.
- Replace the sample CSV with your own historical T20 dataset and retrain.

The application runs on your computer. No API key, external prediction service or hosted website is needed.

## Project layout

```text
cricket-match-outcome-prediction/
├── app.py                         # Streamlit interface
├── train.py                       # Training command-line entry point
├── requirements.txt               # Pinned Python dependencies
├── pytest.ini                     # Test configuration
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml                 # Green UI theme
├── .vscode/
│   └── settings.json               # Local environment and test settings
├── cricket_predictor/
│   ├── __init__.py
│   ├── data.py                     # CSV validation and cleaning
│   ├── features.py                 # Earlier-match-only historical features
│   ├── modeling.py                 # Splits, pipelines, evaluation and saving
│   └── predict.py                  # Artifact loading and inference
├── data/
│   ├── sample_matches.csv         # 720 reproducible synthetic matches
│   └── sample_format.csv          # Small example of the CSV schema
├── scripts/
│   └── generate_sample_data.py     # Deterministic sample-data generator
├── artifacts/
│   ├── best_model.joblib           # Saved pipeline, metadata and clean history
│   ├── metrics.json               # Evaluation and cleaning reports
│   ├── model_comparison.csv        # Scores for all three candidates
│   ├── model_comparison.png        # Accuracy comparison chart
│   └── confusion_matrix.png        # Selected model's held-out test results
├── tests/                         # Data, features, modeling and UI checks
└── docs/
    ├── design.md
    └── implementation-plan.md
```

## Set up in VS Code

Use **Python 3.10–3.12; Python 3.12 is recommended**. The dependencies are pinned for reproducible setup. Install Python, [Visual Studio Code](https://code.visualstudio.com/) and its Microsoft Python extension first.

1. Extract the project ZIP, if supplied.
2. In VS Code, choose **File → Open Folder** and open `cricket-match-outcome-prediction`, the folder containing `app.py`.
3. Choose **Terminal → New Terminal**, then use the commands for your operating system below.
4. Open the Command Palette with **Ctrl+Shift+P** (Windows/Linux) or **Cmd+Shift+P** (macOS). Run **Python: Select Interpreter** and choose the project's `.venv` interpreter. If necessary, use **Enter interpreter path** and select `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux.

Run all commands from this project folder. Using `python -m ...` keeps package installation, Streamlit and tests in the same Python environment.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

If PowerShell blocks activation, you can run the environment's Python directly without changing execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If you installed Python 3.10 or 3.11 instead, substitute that version in the first command.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

Use `python3.10` or `python3.11` in the first command if that is the supported version you installed. On Linux, your distribution may require its `python3-venv` package to create a virtual environment.

Streamlit prints a local address, normally **http://localhost:8501**. Open it in your browser. Press **Ctrl+C** in the terminal to stop the app.

A trained sample artifact is included in the deliverable. Retraining with `python train.py` after installing the pinned requirements is recommended so the saved model matches your Python environment. If no artifact exists, the app also offers **Train demo models**; it only trains when you click that button.

## Use the app

1. In **Predict**, choose two different teams and a venue.
2. Set the toss winner and choose whether they bat or field first.
3. Choose a match date **strictly after the saved history's last date**.
4. Select **Predict match** to see the predicted winner, both estimated probabilities and the recent-history inputs.
5. Open **Model performance** for the candidate comparison, confusion matrix, split dates and cleaning report. **Data guide** explains the format and provides a sample CSV download.

The saved model is fitted on all supplied history, so both the UI and Python API reject dates on or before that history's final date. This prevents presenting a training-period prediction as a new forecast. Changing the future date does not invent new team form; add completed matches to the CSV and retrain to update history.

## CSV format

Use one row per completed T20 match with this header:

```csv
match_id,date,team1,team2,venue,toss_winner,toss_decision,winner,team1_runs,team1_wickets,team2_runs,team2_wickets
demo_001,2025-01-01,Falcons,Tigers,Central Ground,Falcons,bat,Falcons,168,6,159,8
demo_002,2025-01-03,Tigers,Rangers,River Park,Rangers,field,Rangers,151,9,154,5
```

These two rows illustrate the format. Training requires at least **30 usable matches across 5 distinct dates**, with both team1 and team2 wins in the training period. A substantially larger, representative dataset is recommended for meaningful evaluation.

| Column | Meaning |
| --- | --- |
| `match_id` | Unique, non-empty match identifier. |
| `date` | Match date in `YYYY-MM-DD` format. |
| `team1`, `team2` | Two distinct, non-empty team names. Keep spelling consistent across rows. |
| `venue` | Ground or stadium name. |
| `toss_winner` | Exactly one of the two participating teams. |
| `toss_decision` | `bat` or `field`, from the toss winner's perspective. |
| `winner` | Exactly one of the participating teams. |
| `team1_runs`, `team2_runs` | Runs scored by the named team, not necessarily the first/second batting innings. |
| `team1_wickets`, `team2_wickets` | Wickets lost by the named team, from 0 to 10. |
| `data_source` *(optional)* | Data provenance. Use `synthetic` for generated examples. |

The cleaner trims text, parses dates and numeric scores, removes duplicate rows and reports excluded records. Invalid teams, toss details or dates and unusable outcomes such as ties, abandoned games and no results are excluded. Runs must be whole numbers from 0 to 500 and wickets whole numbers from 0 to 10; invalid or missing numeric scores become missing values. The four score columns can be omitted entirely, although including them makes historical averages more useful. Historical averages use available scores; defaults handle absent history. Conflicting records with the same `match_id` or duplicate column names raise an error so they can be corrected instead of silently selecting one.

### Train on your own dataset

Save your CSV as `data/your_matches.csv`, then run:

```bash
python train.py --data data/your_matches.csv --output artifacts
```

Use `--window 10` to calculate recent form from up to ten previous matches, or `python train.py --help` to see all options.

This replaces the saved model and reports in `artifacts/`. Reload the app afterward. Teams, venues, metrics and history dates come from the newly saved artifact. Mark generated data as synthetic; do not label artificial scores as historical results.

### Regenerate the demo data

The bundled sample uses 720 matches and random seed 42:

```bash
python scripts/generate_sample_data.py --output data/sample_matches.csv --matches 720 --seed 42
python train.py
```

The same generator arguments reproduce the same sample. The sample embeds simple team, venue and toss effects for an educational demonstration; those assumptions are not claims about professional cricket.

## How the model works

### Inputs and historical features

The categorical inputs are `team1`, `team2`, `venue`, `toss_winner` and `toss_decision`. For each team, the model also uses:

- Recent win rate: wins divided by previous matches in the window.
- Average runs scored in previous matches.
- Average wickets lost in previous matches.

By default the window is the **last five matches on strictly earlier dates**, whether a team appeared as `team1` or `team2`. Fewer matches are used when fewer are available. With no usable history, the defaults are a 0.5 win rate, 150 runs and 6 wickets. Matches played on the same day never contribute to each other's features because the CSV does not provide completion times.

**The current match's final scores and winner never enter its own features.** Its result becomes available only to later dates. Scores in the CSV are historical observations, not fields the user must know for the match being predicted. The date controls which history is eligible; it is not a numerical model input. The target is `1` when `team1` wins and `0` when `team2` wins.

Categorical values are one-hot encoded with unknown categories supported. Numeric features are imputed and standardized inside fitted scikit-learn pipelines, so preprocessing is learned only from the relevant fitting split. The Python API accepts unseen teams or venues and returns warnings; unseen teams use the history defaults. The UI offers known values from the saved dataset.

### Selection, evaluation and the saved model

1. Sort the cleaned matches chronologically and split distinct dates into approximately **60% training, 20% validation and 20% test**. No date crosses a boundary. With several matches per date, the percentages of match rows may differ.
2. Fit Logistic Regression, Random Forest and XGBoost on the training portion. Compare validation accuracy and validation log loss.
3. Select the highest validation accuracy; if accuracies tie, prefer lower validation log loss.
4. Refit each candidate on training + validation data and evaluate it on the held-out test portion. Test scores do not change the model selected in step 3.
5. Refit the selected model architecture on all available history and save it for future inference.

Test features use rolling history: a previously completed test match can contribute to the features of a test match on a later date. This represents making predictions in sequence as results arrive. Same-day results remain excluded.

The reports include a majority-class baseline to put accuracy in context. The confusion matrix is ordered `[0, 1]`: **Team 2 wins**, then **Team 1 wins**. Rows are actual outcomes and columns are predictions. Accuracy is the fraction of correct winners; lower log loss means the assigned probabilities better match the outcomes.

The saved pipeline has been refitted on more data than the pipelines measured in the test comparison. The displayed test accuracy describes the evaluation procedure, not an additional test of the final all-history refit. Synthetic data, a small or biased real dataset, changing team lineups and conditions, missing player information, and uncalibrated probabilities all limit conclusions about future matches.

## Generated artifacts

| File | Contents |
| --- | --- |
| `artifacts/best_model.joblib` | Selected fitted pipeline, metadata, clean history and evaluation metrics needed for inference. |
| `artifacts/metrics.json` | Validation/test results, selected model, baseline, confusion matrix, split ranges and cleaning report. |
| `artifacts/model_comparison.csv` | All three candidates' validation and test accuracy/log loss. |
| `artifacts/model_comparison.png` | Model accuracy chart. |
| `artifacts/confusion_matrix.png` | Selected model's test confusion matrix. |

The metadata records the model name, history date range, teams, venues, feature window, synthetic-data flag, training timestamp and library versions. Model persistence uses joblib/pickle: **only load artifacts you created or trust**, because loading such files can execute code.

## Use the prediction API

Run this from the project folder after training, or save it as a Python script:

```python
from datetime import date, timedelta
from cricket_predictor.predict import load_artifact, predict_match

artifact = load_artifact("artifacts/best_model.joblib")
next_date = date.fromisoformat(artifact["metadata"]["history_end"]) + timedelta(days=1)
result = predict_match(
    artifact,
    team1="India",
    team2="Australia",
    venue="Wankhede Stadium",
    toss_winner="India",
    toss_decision="bat",
    match_date=next_date,
)
print(result["winner"], f"{result['win_probability']:.1%}")
print(result["probabilities"])
```

## Run the checks

With your project virtual environment active:

```bash
python -m pytest -q
```

Tests cover cleaning and invalid inputs, chronological splits, strictly earlier-date features, model persistence, probability mapping, and Streamlit interactions. To rebuild the evaluation reports, run `python train.py` again.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` or `streamlit` not found | Select the project's `.venv` interpreter, install `requirements.txt`, and use `python -m streamlit run app.py`. |
| Python/dependency installation fails | Use Python 3.10–3.12, preferably 3.12, and upgrade pip. A newer unsupported Python may not have compatible binary wheels. |
| XGBoost reports `libomp.dylib` missing on macOS | Install the OpenMP runtime with `brew install libomp`, then retry in the same environment. |
| XGBoost reports `libgomp.so.1` missing on Linux | Install your distribution's OpenMP runtime, for example `sudo apt-get install libgomp1` on Debian/Ubuntu. |
| A bundled model cannot be loaded or reports version differences | Reinstall the pinned requirements and run `python train.py` to recreate the artifact locally. |
| CSV training fails | Check the required header, valid participating teams and toss values, consistent dates, duplicate IDs, and enough rows/dates for each split. Read the printed error and cleaning report. |
| Date is unavailable in the app | Predictions must be for a date after the final stored history date. Evaluate older matches through the chronological training workflow. |
| Port 8501 is in use | Run `python -m streamlit run app.py --server.port 8502`. |

The app expects its model at `artifacts/best_model.joblib`. If you train with a different `--output` folder, use the Python API with that path or retrain into `artifacts` to use the app.
