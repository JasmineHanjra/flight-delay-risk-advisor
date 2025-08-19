Flight Delay Risk Advisor

A Streamlit application that estimates whether a U.S. domestic flight will arrive ≥ 15 minutes late and suggests lower-risk alternatives (time of day and airline) for the same route. The model is intentionally interpretable, and the UI is designed for non-technical users.

Overview

• Problem: Historical delay statistics are hard to turn into concrete choices for a specific route, date, time, and carrier.
• Goal: Provide a per-flight delay probability and practical, lower-risk alternatives that a traveler can act on.
• Approach: Train an interpretable scikit-learn logistic-regression pipeline on a sampled portion of the U.S. DOT/BTS On-Time Performance dataset with well-understood features (route, carrier, month, weekday, departure hour, distance).

Screenshots

- Prediction card  
  <img width="1781" height="857" alt="image" src="https://github.com/user-attachments/assets/6ced525e-5c92-4de9-b6df-c6aa26612acc" />

- Safer options by time of day  
  <img width="1826" height="680" alt="image" src="https://github.com/user-attachments/assets/710b1587-55a4-4229-a76f-32d8c3deb029" />

- Safer options by airline  
  <img width="1776" height="646" alt="image" src="https://github.com/user-attachments/assets/7f468820-8271-4165-9f78-a23ae194ec8d" />

How to Run (summary)

Create and activate a Python virtual environment.

Install dependencies from requirements.txt.

Provide data by either placing a file named airline_2m.csv next to app.py or by setting the environment variable FLIGHTS_CSV to the full path of your dataset.

Launch the app with Streamlit and open the local URL in your browser.

Large datasets and virtual environments are intentionally excluded from version control via .gitignore.

Data

Source: U.S. Bureau of Transportation Statistics (BTS) On-Time Performance, 1987–2020+.
Target definition: ARR_DEL15 equals 1 if arrival delay is at least 15 minutes, else 0. When ARR_DEL15 is missing, it is derived from available arrival delay fields when possible.
Sampling: The app samples rows from the CSV to keep training fast on a laptop while preserving signal. The random seed is fixed for repeatability.

Features Used

Origin (IATA)
Destination (IATA)
Operating carrier code (two letters)
Month (1–12)
Day of week (1–7)
Scheduled departure hour (derived from HHMM)
Distance (miles)

Categorical features are one-hot encoded. Numeric features are imputed and scaled. Feature choices prioritize interpretability and broad availability across years.

Model

Algorithm: LogisticRegression with class_weight set to “balanced” to address class imbalance.
Pipeline: ColumnTransformer for preprocessing followed by logistic regression.
Split: Train/test split (for example, 80/20), with metrics reported on the held-out test set.

Rationale: Logistic regression is fast, robust, and provides calibrated probabilities that non-technical users can understand. The emphasis is decision support rather than leaderboard performance.

Interpreting Results

Estimated delay risk (for example, 28.1%): The probability that the arrival delay will be at least 15 minutes for the given inputs. Interpreted over many comparable flights, roughly that fraction would be delayed.

Status label: “Likely ON-TIME” or “Likely DELAYED,” based on a decision threshold (default 0.50). The threshold can be adjusted in an Advanced section if a more conservative or more liberal stance on delays is preferred.

Internal test accuracy (for example, 80.2%): Accuracy on the held-out split of the sampled dataset. This is a global performance summary, not a guarantee for any single flight.

Safer options: The app shows lower-risk alternatives by time of day (departure windows) and by airline for the same route and time window, so a user can pivot to safer choices.

Why many cases may read ON-TIME: Delays are often the minority class; probabilities can cluster below 0.5 even with class balancing. Risk typically increases in winter months, on Friday/Sunday evenings, and on busy hub pairs.

Limitations: The model relies on historical behavior only. It does not incorporate live weather, crew constraints, or air traffic initiatives. Treat outputs as a risk advisor rather than a real-time delay prediction.

Evaluation

Reported on the held-out test split:
• Accuracy: overall correctness of ON-TIME versus DELAYED labels.
• (Optional) Precision and Recall for the DELAYED class: helpful if prioritizing catching delays versus minimizing false alarms.
• Calibration: logistic regression generally produces well-calibrated probabilities.

Changing the decision threshold alters the label but not the underlying probability.

Reproducibility

Dependencies are pinned in requirements.txt.
Sampling uses a fixed random seed.
The dataset path is provided via the environment variable FLIGHTS_CSV or by placing airline_2m.csv next to app.py.
A small demo sample can be used to run the app without large files.

Project Structure

app.py — Streamlit UI and training/inference pipeline
requirements.txt — Dependencies
make_sample.py — Optional sample generator for demos
.streamlit/config.toml — Theme and layout
README.md — This document
.gitignore — Excludes large data, models, and virtual environments

Roadmap

Add an optional gradient-boosted model with calibration and side-by-side metrics.
Provide route-specific dashboards and reliability plots.
Enrich features with historical weather summaries.
Export a shareable trip summary (PDF).
Cache a small parquet sample for instant startup.
