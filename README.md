<h1 align="center">Flight Delay Risk Advisor</h1>
<p align="center"><b>Interpretable delay risk estimates and safer alternatives by time of day and airline</b></p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-UI-red.svg">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-Logistic%20Regression-orange.svg">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey.svg">
</p>

---

## At a Glance

- **Purpose** — Turn a traveler’s inputs (route, airline, date, time) into an actionable **probability of arriving ≥ 15 minutes late**, plus **lower-risk alternatives**.
- **Audience** — Non-technical users; designed for clarity and decision support.
- **Model** — Interpretable scikit-learn logistic regression; balanced classes; calibrated probabilities.

---

## What the App Delivers

- **Estimated delay risk** (e.g., 28.1%): chance that your flight arrives ≥ 15 minutes late for the given inputs.  
- **Status label**: “Likely ON-TIME” or “Likely DELAYED” (threshold defaults to 0.50; adjustable in an Advanced panel).  
- **Internal test accuracy** (e.g., 80.2%): held-out performance on the sampled dataset.  
- **Safer options**: side-by-side risk by **time of day** and by **airline** for the same route.

> Notes: Delays are often the minority class; probabilities can cluster below 0.5. Risk typically rises in winter months, on Friday/Sunday evenings, and on busy hub pairs. This is a historical **risk advisor**, not a live nowcast.

---

## Screenshots

<p align="center">
  <img width="1791" height="841" alt="image" src="https://github.com/user-attachments/assets/6698d6d7-38eb-4cf9-acbd-ce5e5d30a333" width="780" alt="Prediction card">
</p>

<p align="center">
  <img width="1781" height="655" alt="image" src="https://github.com/user-attachments/assets/8b1bb25a-f95e-4b03-aaec-8948c2360594" width="780" alt="Safer options by time of day">
</p>

<p align="center">
  <img width="1747" height="666" alt="image" src="https://github.com/user-attachments/assets/98f2f538-4141-4822-a812-3197ed9e59e3" width="780" alt="Safer options by airline">
</p>


---

## How It Works

1. **Data** — U.S. DOT/BTS On-Time Performance (1987–2020+). The app samples rows from a large CSV for fast local training.
2. **Features (kept interpretable)** — `ORIGIN`, `DEST`, `OP_CARRIER`, `MONTH`, `DAY_OF_WEEK`, `DEP_HOUR` (from scheduled HHMM), `DISTANCE`.
3. **Target** — `ARR_DEL15` (1 if arrival delay ≥ 15 minutes; derived when only delay minutes are present).
4. **Pipeline** — One-hot encode categoricals; impute/scale numeric; logistic regression with `class_weight="balanced"`.
5. **Output** — A calibrated probability, a clear label, and ranked lower-risk alternatives.

---

## Running the App

- Create and activate a Python virtual environment, then install dependencies from `requirements.txt`.  
- Provide data by either placing a file named `airline_2m.csv` next to `app.py` **or** setting an environment variable `FLIGHTS_CSV` to your dataset path.  
- Launch Streamlit and open the local URL shown in the terminal.

*(Large CSVs and virtual environments are intentionally excluded from version control.)*

---

## Interpreting Results (Detailed)

| Element | Meaning | Practical use |
|---|---|---|
| Estimated delay risk | Probability of ≥15 min arrival delay for your inputs | Compare options; lower is safer |
| Status label | “Likely ON-TIME/DELAYED” from a threshold (default 0.50) | Quick read; adjust threshold if you prefer conservative alerts |
| Safer by time | Risk vs. departure windows for the same route/airline | Pick an earlier or calmer window |
| Safer by airline | Risk vs. carrier for the same route/time | Switch carriers when feasible |
| Accuracy | Held-out test accuracy on the sampled data | Global quality signal; not a per-flight guarantee |

Limitations: Historical only; no same-day weather/ATC/crew. Accuracy varies by route/season/time.

---

## Data & Reproducibility

- **Source** — U.S. Bureau of Transportation Statistics (BTS) On-Time Performance.  
- **Sampling** — Fixed random seed for repeatable subsamples; keeps training under a minute on a laptop.  
- **Portability** — Dataset path via `FLIGHTS_CSV` or `airline_2m.csv` next to `app.py`.  
- **Dependencies** — Pinned in `requirements.txt`.

---

## Project Structure

- `app.py` — Streamlit UI and training/inference pipeline  
- `requirements.txt` — Dependencies  
- `make_sample.py` — Optional small-sample generator for demos  
- `.streamlit/config.toml` — Theme and layout  
- `README.md` — This document  
- `.gitignore` — Excludes large data, models, and environments

---

## Roadmap (Selected)

- Optional gradient-boosted model with calibration and side-by-side metrics  
- Route-specific dashboards and reliability plots  
- Historical weather enrichment  
- Exportable trip summary (PDF)  
- Cached parquet sample for instant startup

---


MIT License. Suitable for academic and personal use.
