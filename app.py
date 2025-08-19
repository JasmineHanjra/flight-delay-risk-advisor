# Flight Delay Risk Advisor — polished UI for non-technical users
from __future__ import annotations
import os
from typing import Tuple, Dict, List
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ========================== CONFIG ==========================
import os
DATA_PATH = os.getenv("FLIGHTS_CSV", "airline_2m.csv")  # uses env var or a local file next to app.py
ROWS_TO_SAMPLE = 200_000
CLASSIFY_THRESHOLD = 0.50
# ============================================================

st.set_page_config(page_title="Flight Delay Risk Advisor", layout="wide")
st.markdown("""
<style>
  .block-container { padding-top: 1rem; padding-bottom: 1.2rem; }
  h1,h2,h3 { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
  .muted { color:#6b7280; font-size:.92rem; }
  .card { background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:14px 16px; }
  .soft { background:#f6f8fa; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; }
  .subtle { color:#374151; }
  .tight { margin-top:-6px; }
</style>
""", unsafe_allow_html=True)

# ---------------------- helpers ----------------------
def to_scalar(x) -> float:
    arr = np.asarray(x)
    return float(arr.ravel()[0]) if arr.size else float("nan")

def fmt_pct(p):
    p = to_scalar(p)
    return "—" if np.isnan(p) else f"{p*100:.1f}%"

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

def normalize_bts_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    alias = {
        "FLIGHTDATE": "FL_DATE",
        "DAYOFWEEK": "DAY_OF_WEEK",
        "CRSDEPTIME": "CRS_DEP_TIME",
        "DEPTIME": "DEP_TIME",
        # Only map Reporting_Airline to OP_CARRIER (avoid duplicate with IATA_CODE_REPORTING_AIRLINE)
        "REPORTING_AIRLINE": "OP_CARRIER",
        "ARRDEL15": "ARR_DEL15",
        "ORIGINCITYNAME": "ORIGIN_CITY_NAME",
        "DESTCITYNAME": "DEST_CITY_NAME",
        "CRSARRTIME": "CRS_ARR_TIME",
        "ARRTIME": "ARR_TIME",
    }
    df.rename(columns=lambda c: alias.get(c, c), inplace=True)
    return ensure_unique_columns(df)

def parse_dep_hour(v):
    if pd.isna(v): return np.nan
    try:
        x = int(float(np.asarray(v).ravel()[0]))
    except Exception:
        return np.nan
    return x if 0 <= x <= 23 else (x // 100) % 24

def make_demo(n_rows=6000, seed=42):
    rng = np.random.default_rng(seed)
    origins = ["JFK","LGA","EWR","BOS","DCA","ATL","ORD","DFW","LAX","SFO"]
    dests   = ["LAX","SFO","SEA","DEN","PHX","ORD","ATL","MIA","DFW","BOS"]
    carriers= ["AA","DL","UA","WN","B6","AS"]
    months  = rng.integers(1,13,size=n_rows)
    dows    = rng.integers(1,8,size=n_rows)
    dep_hr  = rng.integers(0,24,size=n_rows)
    origin  = rng.choice(origins, size=n_rows)
    dest    = rng.choice(dests, size=n_rows)
    carrier = rng.choice(carriers, size=n_rows)
    distance= rng.normal(1500, 600, size=n_rows).clip(200, 3000).astype(int)
    base = 0.10 + 0.02*(np.isin(months,[12,1,2])) + 0.05*((dep_hr>=17)&(dep_hr<=22)) + 0.03*(carrier=="WN")
    base += 0.04*((origin=="EWR")|(origin=="ORD")) + 0.03*((dest=="SFO")|(dest=="LAX")) + 0.02*(dows>=5)
    y = (rng.random(n_rows) < base).astype(int)
    return pd.DataFrame({
        "ORIGIN":origin,"DEST":dest,"OP_CARRIER":carrier,
        "MONTH":months,"DAY_OF_WEEK":dows,"DEP_HOUR":dep_hr,
        "DISTANCE":distance,"ARR_DEL15":y
    })

def read_csv_sample_any_encoding(path: str, usecols: List[str], n_rows: int, seed: int = 42):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        try:
            chunks, need = [], int(n_rows)
            reader = pd.read_csv(
                path, usecols=lambda c: c in usecols,
                chunksize=100_000, encoding=enc, on_bad_lines="skip", engine="python"
            )
            for ch in reader:
                if need <= 0: break
                take = min(len(ch), need)
                chunks.append(ch.sample(n=take, random_state=seed) if take < len(ch) else ch)
                need -= len(chunks[-1])
            if chunks:
                return pd.concat(chunks, ignore_index=True), enc
            df_small = pd.read_csv(path, usecols=lambda c: c in usecols,
                                   encoding=enc, on_bad_lines="skip", engine="python")
            return df_small, enc
        except UnicodeDecodeError:
            continue
    df_fallback = pd.read_csv(path, usecols=lambda c: c in usecols,
                              encoding="latin1", on_bad_lines="skip", engine="python")
    return df_fallback, "latin1 (fallback)"

@st.cache_data(show_spinner=False)
def load_data():
    usecols = [
        "FlightDate","Month","DayOfWeek","Origin","Dest",
        "Reporting_Airline","IATA_CODE_Reporting_Airline",
        "CRSDepTime","DepTime","Distance",
        "Cancelled","OriginCityName","DestCityName",
        "ArrDel15","ArrDelay","ArrDelayMinutes","ArrivalDelay",
        "ArrTime","CRSArrTime",
    ]
    if os.path.exists(DATA_PATH):
        df_raw, enc_used = read_csv_sample_any_encoding(DATA_PATH, usecols, ROWS_TO_SAMPLE, seed=42)
        return df_raw, f"Using local DOT/BTS data (encoding: {enc_used}; {len(df_raw):,} rows)"
    else:
        return make_demo(6000), "Using small built-in demo data"

def prepare_features(df_raw: pd.DataFrame):
    df = normalize_bts_columns(df_raw)
    if "CANCELLED" in df.columns:
        df = df[df["CANCELLED"].fillna(0) == 0]

    def hhmm_to_minutes(x):
        try:
            x = int(float(np.asarray(x).ravel()[0])); return (x // 100) * 60 + (x % 100)
        except Exception:
            return np.nan

    # Label
    y = None
    for c in ["ARR_DEL15", "ARRDEL15"]:
        if c in df.columns:
            y = pd.to_numeric(df[c], errors="coerce").round().clip(0,1).astype("Int64").fillna(0).astype(int); break
    if y is None:
        for c in ["ARR_DELAY","ARRDELAY","ARR_DELAY_MINUTES","ARRDELAYMINUTES","ARRIVAL_DELAY","ARRIVALDELAY"]:
            if c in df.columns:
                y = (pd.to_numeric(df[c], errors="coerce") >= 15).astype(int); break
    if y is None and ("ARR_TIME" in df.columns and "CRS_ARR_TIME" in df.columns):
        arr_m = df["ARR_TIME"].apply(hhmm_to_minutes)
        crs_m = df["CRS_ARR_TIME"].apply(hhmm_to_minutes)
        diff = arr_m - crs_m
        diff = np.where(diff < -600, diff + 24*60, diff)
        y = (pd.to_numeric(diff, errors="coerce") >= 15).astype(int)
    if y is None:
        raise ValueError("Could not derive delay label (need ArrDel15 or ArrDelayMinutes or ArrTime/CRSArrTime).")

    # Features
    dep_source = next((c for c in ["CRS_DEP_TIME", "DEP_TIME", "CRS_DEP_HOUR", "DEP_HOUR"] if c in df.columns), None)
    if dep_source is None:
        raise ValueError("Need CRSDepTime or DepTime to compute departure hour.")
    df["DEP_HOUR"] = df[dep_source].apply(parse_dep_hour)

    if "OP_CARRIER" not in df.columns:
        for c in ["CARRIER","OP_UNIQUE_CARRIER","IATA_CODE_REPORTING_AIRLINE"]:
            if c in df.columns:
                df["OP_CARRIER"] = df[c].astype(str); break

    if ("MONTH" not in df.columns or "DAY_OF_WEEK" not in df.columns) and "FL_DATE" in df.columns:
        dtcol = pd.to_datetime(df["FL_DATE"], errors="coerce")
        if "MONTH" not in df.columns: df["MONTH"] = dtcol.dt.month
        if "DAY_OF_WEEK" not in df.columns: df["DAY_OF_WEEK"] = (dtcol.dt.weekday + 1).astype(int)

    need = ["ORIGIN","DEST","DISTANCE","MONTH","DAY_OF_WEEK","OP_CARRIER","DEP_HOUR"]
    for col in need:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = ensure_unique_columns(df)
    X = df[need].copy()
    X = ensure_unique_columns(X)
    city_map = {}
    if "ORIGIN_CITY_NAME" in df.columns:
        city_map.update(dict(zip(df["ORIGIN"], df["ORIGIN_CITY_NAME"])))
    if "DEST_CITY_NAME" in df.columns:
        city_map.update(dict(zip(df["DEST"], df["DEST_CITY_NAME"])))
    return X, y.astype(int), {"city_names": city_map}

@st.cache_resource(show_spinner=False)
def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    if not X.columns.is_unique:
        dupes = [c for c in X.columns[X.columns.duplicated()]]
        raise ValueError(f"Non-unique columns in features: {dupes}")
    num_cols = ["MONTH","DAY_OF_WEEK","DEP_HOUR","DISTANCE"]
    cat_cols = ["ORIGIN","DEST","OP_CARRIER"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])
    clf = LogisticRegression(max_iter=200, solver="liblinear", C=1.0)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    return pipe

CARRIER_NAMES = {
    "AA":"American", "DL":"Delta", "UA":"United", "WN":"Southwest", "B6":"JetBlue",
    "AS":"Alaska", "NK":"Spirit", "F9":"Frontier", "HA":"Hawaiian", "G4":"Allegiant",
    "OO":"SkyWest", "YX":"Republic", "YV":"Mesa", "MQ":"Envoy/American Eagle",
}
def carrier_label(code: str) -> str:
    name = CARRIER_NAMES.get(code, "")
    return f"{name} ({code})" if name else code

BUCKETS = {
    "Early morning (5–8)": 6,
    "Morning (9–11)": 10,
    "Afternoon (12–16)": 14,
    "Evening (17–20)": 18,
    "Night (21–23)": 21,
    "Red-eye (0–4)": 1,
}

# ---------------------- data/model ----------------------
df_raw, source_msg = load_data()
X_all, y_all, info = prepare_features(df_raw)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
model = train_model(X_tr, y_tr)
quick_acc = accuracy_score(y_te, (model.predict_proba(X_te)[:,1] >= CLASSIFY_THRESHOLD).astype(int))

@st.cache_data(show_spinner=False)
def make_route_distance_lookup(X: pd.DataFrame) -> Dict[tuple, float]:
    g = X.groupby(["ORIGIN","DEST"])["DISTANCE"].median()
    return {k: float(v) for k,v in g.items()}

route_dist = make_route_distance_lookup(X_all)
global_median_distance = float(np.median(X_all["DISTANCE"]))

# ---------------------- HERO ----------------------
st.title("Flight Delay Risk Advisor")
st.markdown(
    f"<div class='soft subtle'>Enter your route and airline. We estimate the chance of a ≥15-minute delay and suggest safer options."
    f"<div class='muted tight'>{source_msg}. No settings required.</div></div>",
    unsafe_allow_html=True
)

# ---------------------- INPUTS ----------------------
flight_no = st.text_input("Flight # (optional)", placeholder="e.g., AA123")

orig_codes = sorted(X_all["ORIGIN"].dropna().unique())
dest_codes = sorted(X_all["DEST"].dropna().unique())
carriers = sorted(X_all["OP_CARRIER"].dropna().unique())

city_names = info["city_names"]
def pretty_airport(code: str) -> str:
    city = city_names.get(code, "")
    return f"{city} ({code})" if city else code

origin_opts = [pretty_airport(c) for c in orig_codes]
dest_opts   = [pretty_airport(c) for c in dest_codes]
carrier_opts= [carrier_label(c) for c in carriers]
label_to_code = {carrier_label(c): c for c in carriers}

c1, c2 = st.columns(2)
sel_origin_label = c1.selectbox("From", origin_opts, index=0, help="Departure airport")
sel_dest_label   = c2.selectbox("To", dest_opts, index=min(1, len(dest_opts)-1), help="Arrival airport")

sel_carrier_label = st.selectbox("Airline", carrier_opts, index=0, help="Airline operating the flight")
sel_carrier = label_to_code[sel_carrier_label]

colD, colT = st.columns([1,1])
sel_date = colD.date_input("Departure date", dt.date.today())
sel_time = colT.time_input("Departure time", dt.time(10, 0))

def code_from_label(lbl: str) -> str:
    if "(" in lbl and lbl.endswith(")"):
        return lbl.split("(")[-1].split(")")[0]
    return lbl

sel_origin = code_from_label(sel_origin_label)
sel_dest   = code_from_label(sel_dest_label)

if sel_origin == sel_dest:
    st.warning("Origin and destination are the same. Please choose different airports.")
    st.stop()

sel_month = sel_date.month
sel_dow   = sel_date.weekday() + 1
sel_hour  = sel_time.hour
dist = route_dist.get((sel_origin, sel_dest), global_median_distance)

row = {"ORIGIN": sel_origin, "DEST": sel_dest, "OP_CARRIER": sel_carrier,
       "MONTH": sel_month, "DAY_OF_WEEK": sel_dow, "DEP_HOUR": sel_hour, "DISTANCE": dist}

# ---------------------- PREDICTION CARD ----------------------
proba = to_scalar(model.predict_proba(pd.DataFrame([row]))[:,1])
label = "Likely DELAYED" if proba >= CLASSIFY_THRESHOLD else "Likely ON-TIME"

st.subheader("Your flight’s delay risk")
with st.container():
    cA, cB, cC = st.columns([1,1,1])
    cA.metric("Estimated risk", fmt_pct(proba))
    cB.metric("Status", label)
    cC.metric("Model accuracy (overall)", f"{quick_acc*100:.1f}%")
st.progress(int(round(proba * 100)),
            text=f"Probability of a ≥15-minute delay: {fmt_pct(proba)}")
st.caption("Lower is better. Try earlier departures or a different airline — see safer options below.")

st.divider()

# ---------------------- SAFER OPTIONS ----------------------
st.subheader("Safer options for your trip")

# By time of day
rows = []
for bucket, hour in BUCKETS.items():
    r = row.copy(); r["DEP_HOUR"] = hour
    p = to_scalar(model.predict_proba(pd.DataFrame([r]))[:,1])
    rows.append({"Time of day": bucket, "Delay risk": p})
bucket_df = pd.DataFrame(rows).sort_values("Delay risk")
best_time = bucket_df.iloc[0]["Time of day"]

left, right = st.columns([1,1])
left.markdown("**Same route & airline — try a different time**")
left.dataframe(
    bucket_df.assign(**{"Delay risk": lambda d: (d["Delay risk"]*100).round(1)}),
    use_container_width=True, hide_index=True
)
fig = px.bar(bucket_df, x="Time of day", y="Delay risk",
             title="Delay risk by time of day", labels={"Delay risk":"Delay risk"})
fig.update_yaxes(tickformat=".0%")
right.plotly_chart(fig, use_container_width=True)

# By airline
alt_rows = []
for c in carriers:
    r = row.copy(); r["OP_CARRIER"] = c
    p = to_scalar(model.predict_proba(pd.DataFrame([r]))[:,1])
    alt_rows.append({"Airline": carrier_label(c), "Delay risk": p})
alt_df = pd.DataFrame(alt_rows).sort_values("Delay risk")
best_airline = alt_df.iloc[0]["Airline"]
if best_airline == carrier_label(sel_carrier) and len(alt_df) > 1:
    best_airline = alt_df.iloc[1]["Airline"]

st.markdown("**Same route & time — try a different airline**")
st.dataframe(
    alt_df.assign(**{"Delay risk": lambda d: (d["Delay risk"]*100).round(1)}),
    use_container_width=True, hide_index=True
)

st.success(f"Recommendation: consider **{best_time}** and **{best_airline}** for the lowest expected delay risk on this route.")

# ---------------------- ABOUT ----------------------
with st.expander("About this tool"):
    st.write(
        "This app uses historical U.S. DOT/BTS On-Time Performance data sampled locally. "
        "We train a simple, interpretable model (logistic regression) with standard preprocessing. "
        "Accuracy shown above is measured on a held-out split and reflects overall quality, not any single prediction."
    )
