# Verification

Verified on 2026-10-18 with Python 3.12 and `requirements.txt` installed in a clean virtual environment.

- `python -m pytest -q`: **44 passed**, no test warnings.
- `python train.py`: Logistic Regression, Random Forest and XGBoost trained and evaluated successfully; all five output artifacts were generated.
- `python -m pip check`: no broken requirements.
- Streamlit started successfully; `/_stcore/health` returned HTTP **200**, body `ok`.
- Streamlit AppTest covered loading and comparison, prediction probabilities and historical inputs, changed team selections, and missing/corrupt artifact recovery.
- The bundled sample contains 720 synthetic rows and reproduces with seed 42.

The sandbox has a read-only home directory, so verification used writable temporary Matplotlib/cache directories. Ordinary local VS Code setup needs only the README commands.

## Demo training results

These are results on **synthetic data**, not measured accuracy on real cricket matches.

| Model | Validation accuracy | Test accuracy |
| --- | ---: | ---: |
| Logistic Regression | 57.64% | 63.19% |
| Random Forest | 59.03% | 63.89% |
| XGBoost | 59.72% | 63.89% |

XGBoost was selected using validation accuracy. The majority-training-class test baseline was 44.44%. The saved pipeline was then refitted on all 720 matches for future predictions. Full scores, split dates, the confusion matrix and cleaning counts are in `artifacts/metrics.json`.
