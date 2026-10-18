from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cricket_predictor.predict import load_artifact, predict_match

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sample_matches.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"

st.set_page_config(page_title="Cricket Match Predictor", page_icon="🏏", layout="wide")

@st.cache_resource(show_spinner=False)
def cached_artifact(path: str, modified_ns: int) -> dict:
    """Reload after retraining, without reloading on every widget interaction."""
    return load_artifact(path)

def training_prompt() -> None:
    """Training is an explicit user action and never runs on a normal rerun."""
    st.info("Train the three demo models to create your saved prediction pipeline.")
    st.code("python train.py", language="bash")
    st.caption("Or use the button below to train on the bundled synthetic sample CSV.")
    if st.button("Train demo models", type="primary"):
        try:
            from cricket_predictor.modeling import train_models

            with st.spinner("Training Logistic Regression, Random Forest and XGBoost…"):
                train_models(DATA_PATH, ARTIFACTS_DIR)
            cached_artifact.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Training could not finish: {exc}")
            st.caption("Check the CSV and installed dependencies, then run python train.py for details.")

def show_prediction(result: dict, team1: str, team2: str, window: int) -> None:
    st.subheader("Match outlook")
    st.success(f"Predicted winner: {result['winner']}")
    first, second = st.columns(2)
    first.metric(f"{team1} · win probability", f"{result['probabilities'][team1]:.1%}")
    second.metric(f"{team2} · win probability", f"{result['probabilities'][team2]:.1%}")
    st.caption("These are model-estimated probabilities. They are not calibrated probabilities, guarantees, or betting odds.")
    for warning in result.get("warnings", []):
        st.warning(warning)

    features = result["features"].iloc[0]
    st.subheader(f"Recent team history · up to {window} previous matches")
    history = pd.DataFrame(
        {
            "Team": [team1, team2],
            "Recent win rate": [features["team1_recent_form"], features["team2_recent_form"]],
            "Average runs scored": [features["team1_avg_runs"], features["team2_avg_runs"]],
            "Average wickets lost": [features["team1_avg_wickets"], features["team2_avg_wickets"]],
        }
    )
    history["Recent win rate"] = history["Recent win rate"].map(lambda value: f"{value:.0%}")
    st.dataframe(
        history,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Average runs scored": st.column_config.NumberColumn(format="%.1f"),
            "Average wickets lost": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "Only matches on earlier dates contribute. Missing history uses a 50% win rate, "
        "150 runs and 6 wickets; available past scores are averaged when present."
    )
    with st.expander("See the exact model input"):
        st.dataframe(result["features"], hide_index=True, use_container_width=True)


def prediction_tab(artifact: dict) -> None:
    metadata = artifact["metadata"]
    teams = metadata["teams"]
    venues = metadata["venues"]
    if len(teams) < 2 or not venues:
        st.error("The saved dataset needs at least two teams and one venue. Train with a larger CSV.")
        return

    history_end = date.fromisoformat(metadata["history_end"])
    earliest_date = history_end + timedelta(days=1)
    st.subheader("Set up the match")
    st.write("Choose the teams and toss result before play begins.")
    with st.container(border=True):
        first, second = st.columns(2)
        with first:
            team1 = st.selectbox("Team 1", teams, key="team1")
            venue = st.selectbox("Venue", venues, key="venue")
            toss_decision = st.selectbox(
                "Toss decision", ["bat", "field"],
                format_func=lambda value: "Bat first" if value == "bat" else "Field first",
                key="toss_decision",
            )
        with second:
            team2 = st.selectbox("Team 2", [team for team in teams if team != team1], key="team2")
            toss_winner = st.selectbox("Toss winner", [team1, team2], key="toss_winner")
            match_date = st.date_input(
                "Match date", value=max(date.today(), earliest_date), min_value=earliest_date,
                help="The saved model was trained on every match through the history end date.",
                key="match_date",
            )
        st.caption(
            f"Available history ends {history_end:%d %b %Y}. "
            f"Choose {earliest_date:%d %b %Y} or later for a new prediction."
        )
        requested = st.button("Predict match", type="primary", use_container_width=True)

    signature = (MODEL_PATH.stat().st_mtime_ns, team1, team2, venue, toss_winner, toss_decision, str(match_date))
    if requested:
        try:
            with st.spinner("Calculating the match outlook…"):
                result = predict_match(artifact, team1, team2, venue, toss_winner, toss_decision, match_date)
            st.session_state["prediction"] = {"signature": signature, "result": result}
        except (ValueError, TypeError, KeyError) as exc:
            st.session_state.pop("prediction", None)
            st.error(f"Unable to predict this match: {exc}")

    saved = st.session_state.get("prediction")
    if saved and saved["signature"] == signature:
        show_prediction(saved["result"], team1, team2, metadata["feature_window"])
    else:
        st.info("Your match outlook and the historical inputs will appear here after you select Predict match.")


def performance_tab(artifact: dict) -> None:
    metrics = artifact["metrics"]
    st.subheader("Compare the three models")
    st.write("The winner is selected using validation accuracy, with lower validation log loss breaking a tie.")
    selected, test, baseline = st.columns(3)
    selected.metric("Selected model", metrics["selected_model"])
    test.metric("Selected model · test accuracy", f"{metrics['test_accuracy']:.1%}")
    baseline.metric("Majority-class baseline · test accuracy", f"{metrics['baseline_test_accuracy']:.1%}")
    st.caption("Test results are a final held-out comparison and do not decide which model is selected.")

    candidates = pd.DataFrame(metrics["candidates"])
    comparison = candidates.rename(
        columns={
            "name": "Model", "validation_accuracy": "Validation accuracy (%)",
            "validation_log_loss": "Validation log loss", "test_accuracy": "Test accuracy (%)",
            "test_log_loss": "Test log loss",
        }
    )
    for column in ["Validation accuracy (%)", "Test accuracy (%)"]:
        comparison[column] *= 100
    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
        column_config={
            column: st.column_config.NumberColumn(format="%.2f")
            for column in comparison.columns if column != "Model"
        },
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Validation and test accuracy**")
        fig, axis = plt.subplots(figsize=(6, 4))
        try:
            positions = np.arange(len(candidates))
            axis.barh(positions + 0.18, candidates["validation_accuracy"] * 100, height=0.34, label="Validation", color="#94cbb3")
            axis.barh(positions - 0.18, candidates["test_accuracy"] * 100, height=0.34, label="Test", color="#177349")
            axis.set_yticks(positions, candidates["name"])
            axis.set_xlim(0, 100)
            axis.set_xlabel("Accuracy (%)")
            axis.legend(frameon=False, loc="lower right")
            axis.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        finally:
            plt.close(fig)
    with right:
        st.markdown("**Selected model · test confusion matrix**")
        matrix = np.asarray(metrics["test_confusion_matrix"])
        fig, axis = plt.subplots(figsize=(6, 4))
        try:
            axis.imshow(matrix, cmap="Greens", vmin=0)
            labels = ["Team 2 wins", "Team 1 wins"]
            axis.set_xticks([0, 1], labels)
            axis.set_yticks([0, 1], labels)
            axis.set_xlabel("Predicted outcome")
            axis.set_ylabel("Actual outcome")
            for row in range(2):
                for column in range(2):
                    axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                              color="white" if matrix[row, column] > matrix.max() / 2 else "#163629", fontsize=15)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        finally:
            plt.close(fig)

    st.subheader("How the evaluation works")
    st.write(
        "Matches are split in date order into approximately 60% training, 20% validation and 20% test data. "
        "Whole days stay together. Each candidate is first fitted on training data for validation selection, "
        "then refitted on training + validation data for the test comparison. "
        "The selected model is finally refitted on all history and saved for future predictions."
    )
    split_rows = [{"Split": name.title(), "Matches": values["rows"], "First date": values["start"],
                   "Last date": values["end"]} for name, values in metrics["split"].items()]
    st.dataframe(pd.DataFrame(split_rows), hide_index=True, use_container_width=True)
    st.caption(
        "Test features use earlier completed matches, including earlier test matches, as rolling history. "
        "The baseline always chooses the majority training outcome; it is a reference point for model accuracy. "
        "Lower log loss is better."
    )
    with st.expander("Data cleaning report"):
        st.json(metrics["cleaning_report"])


def data_guide_tab(artifact: dict) -> None:
    st.subheader("From a CSV to a match prediction")
    st.write(
        "Each row is a completed T20 match. The model uses team names, venue, toss winner, toss decision, "
        "and recent historical win rates, runs scored and wickets lost. The prediction is made after the toss, before play."
    )
    st.markdown(
        "**CSV format**\n\n"
        "`match_id`, `date`, `team1`, `team2`, `venue`, `toss_winner`, `toss_decision`, "
        "`winner`, `team1_runs`, `team1_wickets`, `team2_runs`, `team2_wickets`.\n\n"
        "Use `YYYY-MM-DD` dates and `bat` or `field` for the toss decision. "
        "The optional `data_source` column can identify synthetic records. "
        "Runs and wickets belong to the named team; wickets mean wickets lost. "
        "Score columns can be omitted when unavailable; historical defaults then apply."
    )
    st.info(
        "A match’s final score and winner are never inputs to its own prediction. "
        "Recent statistics use up to five previous matches by default, from strictly earlier dates. "
        "Same-day matches cannot inform one another."
    )
    st.markdown("**Train on your own CSV**")
    st.code("python train.py --data data/your_matches.csv --output artifacts", language="bash")
    st.write("Reload the app after training. The teams, venues, history dates and metrics update from the new saved model.")
    st.caption(
        "Cleaning removes invalid team, toss, date and unusable-outcome rows, and treats invalid scores as missing. "
        "Conflicting duplicate match IDs must be corrected. See README.md for the complete schema and setup."
    )
    if DATA_PATH.exists():
        st.download_button("Download synthetic sample CSV", DATA_PATH.read_bytes(), "sample_matches.csv", "text/csv")
    with st.expander("Saved model details"):
        st.json(artifact["metadata"])


def main() -> None:
    st.caption("T20 CRICKET · LOCAL MACHINE LEARNING DEMO")
    st.title("🏏 Cricket Match Predictor")
    st.write("Explore a match outlook from the toss and each team’s recent form.")
    if not MODEL_PATH.exists():
        st.warning("No trained model found. The bundled data is synthetic and intended for learning.")
        training_prompt()
        return
    try:
        artifact = cached_artifact(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"The saved model could not be loaded: {exc}")
        st.caption("Install requirements.txt in this environment and retrain the model if library versions differ.")
        training_prompt()
        return

    metadata = artifact["metadata"]
    if metadata["synthetic_data"]:
        st.warning("Synthetic demo data · These results demonstrate the workflow and are not evidence of real-world predictive accuracy.")
    else:
        st.info("Model estimates depend on your historical dataset. Held-out performance does not guarantee future results.")

    with st.sidebar:
        st.header("Your saved model")
        st.markdown(f"**{metadata['model_name']}**")
        st.caption(f"History: {metadata['history_start']} → {metadata['history_end']}")
        st.caption(f"{len(metadata['teams'])} teams · {len(metadata['venues'])} venues")
        st.caption(f"Recent form window: {metadata['feature_window']} matches")
        st.divider()
        st.markdown("**Refresh your model**")
        st.code("python train.py", language="bash")
        st.caption("Training writes a new model and evaluation reports to artifacts/.")

    prediction, performance, guide = st.tabs(["Predict", "Model performance", "Data guide"])
    with prediction:
        prediction_tab(artifact)
    with performance:
        performance_tab(artifact)
    with guide:
        data_guide_tab(artifact)


if __name__ == "__main__":
    main()
