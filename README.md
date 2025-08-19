# Flight Delay Risk Advisor

A Streamlit application that estimates whether a U.S. domestic flight will arrive **≥ 15 minutes late** and suggests **lower-risk alternatives** (time of day and airline) for the same route. The model is intentionally **interpretable** and the UI targets non-technical users.

## Overview

- **Problem**: Travelers can’t easily translate historical performance into actionable choices.
- **Goal**: Provide a per-flight **delay probability** and practical alternatives that reduce risk.
- **Approach**: Train a logistic-regression pipeline on sampled U.S. DOT/BTS On-Time Performance data with well-understood features (route, carrier, month, weekday, departure hour, distance).

---

## Screenshots

> Replace file names below with yours. Place images under `assets/` in the repo and update paths.

- Prediction card  
  <img width="1781" height="857" alt="image" src="https://github.com/user-attachments/assets/6ced525e-5c92-4de9-b6df-c6aa26612acc" />

- Safer options by time of day  
  <img width="1826" height="680" alt="image" src="https://github.com/user-attachments/assets/710b1587-55a4-4229-a76f-32d8c3deb029" />

- Safer options by airline  
  <img width="1776" height="646" alt="image" src="https://github.com/user-attachments/assets/7f468820-8271-4165-9f78-a23ae194ec8d" />


---

## How to Run

### Windows (PowerShell)

```powershell
# 1) Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Point the app to your CSV
# Option A: put airline_2m.csv next to app.py
# Option B: set an environment variable to a custom path
setx FLIGHTS_CSV "C:\path\to\airline_2m.csv"  # reopen terminal after setx
# For the current shell only:
# $env:FLIGHTS_CSV="C:\path\to\airline_2m.csv"

# 4) Launch
python -m streamlit run app.py
macOS / Linux

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: point to dataset
export FLIGHTS_CSV="/path/to/airline_2m.csv"

streamlit run app.py
If no CSV is found, the app can operate in a small demo mode so reviewers can try it without large files.

Data
Source: U.S. Bureau of Transportation Statistics (BTS) On-Time Performance dataset (1987–2020+).

Target definition: ARR_DEL15 (1 if arrival delay ≥ 15 minutes, else 0). If not present, it is derived from arrival delay minutes.

Sampling: The app samples rows from the CSV for fast local training. This keeps training under a minute on a laptop while retaining signal.

Large datasets (CSV) are intentionally excluded from git. The app uses FLIGHTS_CSV or a file named airline_2m.csv in the project folder.

Features
All features are chosen for interpretability and availability across years:

ORIGIN (IATA)

DEST (IATA)

OP_CARRIER (two-letter carrier code)

MONTH (1–12)

DAY_OF_WEEK (1–7)

DEP_HOUR (derived from scheduled departure time)

DISTANCE (miles)

Categorical variables are one-hot encoded; numeric variables are imputed and scaled.

Model
Algorithm: LogisticRegression (scikit-learn) with class_weight="balanced" to address class imbalance.

Pipeline: ColumnTransformer for preprocessing → logistic regression.

Split: Train/validation split (e.g., 80/20). Metrics reported on held-out split.

Rationale: Logistic regression is fast, robust, and yields calibrated probabilities that non-technical users can interpret.

Interpreting Results
The app outputs:

Estimated Delay Risk (e.g., 28.1%)
Probability that the arrival delay will be ≥ 15 minutes for the given inputs.
Interpretation: If an identical flight were taken many times under similar conditions, approximately that fraction would be delayed.

Status Label

“Likely ON-TIME” or “Likely DELAYED” using a threshold (default 0.50).

The threshold can be adjusted in the “Advanced” expander if a user prefers a more conservative stance on delays.

Internal Test Accuracy (e.g., 80.2%)
Accuracy on the held-out split of the sampled training data. This is overall performance, not a guarantee for any single flight.

Safer Options

By time of day: risk estimates for departure windows on the same route and airline.

By airline: risk estimates for other carriers on the same route and time window.

Notes:

Delays are often the minority class; probabilities may cluster below 0.5. Winter months, Friday/Sunday evenings, and busy hubs typically produce higher risk.

The model uses historical behavior only; it does not ingest live weather or operational constraints.

Evaluation
Metrics computed on the held-out test split:

Accuracy: overall correctness of ON-TIME/DELAYED labels.

(Optional) Precision/Recall for DELAYED: configurable in code; helpful when emphasizing catching delays vs. minimizing false alarms.

Probability calibration: logistic regression provides probabilities that are generally well-calibrated.

If you adjust the decision threshold in the UI, the label changes but the underlying probability does not.

Reproducibility
Dependencies: pinned in requirements.txt.

Sampling: fixed random seed for repeatable subsamples.

Portability: dataset path via environment variable or local file adjacent to app.py.

Optional helper:

make_sample.py can produce a small CSV sample for quick demos. This file is not required for normal operation.

Project Structure

├─ app.py                      # Streamlit UI + training/inference pipeline
├─ requirements.txt            # Dependencies
├─ make_sample.py              # (optional) demo sampler
├─ .streamlit/
│   └─ config.toml             # Theme and layout
├─ README.md
└─ .gitignore                  # Excludes venv, large data, model artifacts

Limitations and Scope
Historical data cannot capture day-of-travel disruptions (weather, ATC initiatives, crew legality, etc.).

Risk is route- and season-dependent; the model favors interpretability over complex feature interactions.

Accuracy varies by route and time window; global accuracy is a summary metric.

Roadmap (Selected)
Add optional gradient-boosted trees with calibration and side-by-side metrics.

Route-specific evaluation dashboards and reliability plots.

Weather enrichment using historical station summaries.

Exportable trip summary (PDF).
