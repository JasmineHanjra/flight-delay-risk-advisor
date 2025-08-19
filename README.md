# Flight Delay Risk Advisor

Streamlit app that estimates the chance a flight will be **delayed ≥15 minutes** and suggests safer options (time-of-day / airline), trained on sampled U.S. DOT/BTS On-Time data.

## Quickstart
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: point to your CSV (reopen terminal after setx)
setx FLIGHTS_CSV "C:\path\to\airline_2m.csv"

python -m streamlit run app.py
