import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="VoltIQ | EV Charging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1020; color: #e8edf7; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #101a2d 0%, #0d1424 100%); border-right: 1px solid #24324a; }
[data-testid="stSidebar"] * { color: #e8edf7; }
.block-container { padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1500px; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.03em; }
.hero { padding: 1.5rem 1.7rem; border-radius: 22px; background: radial-gradient(circle at 90% 10%, rgba(43, 224, 157, .18), transparent 30%), linear-gradient(120deg, #15233a, #10192b 65%, #112b2b); border: 1px solid #263b50; margin-bottom: 1.2rem; }
.eyebrow { color: #44e0a7; font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
.hero h1 { font-size: clamp(2rem, 4vw, 3.2rem); margin: .25rem 0 .4rem 0; color: #f5f8ff; }
.hero p { color: #aab9ce; max-width: 760px; margin-bottom: 0; }
.kpi { background: linear-gradient(145deg, #151f33, #111a2b); border: 1px solid #27364e; border-radius: 18px; padding: 1rem 1.1rem; min-height: 120px; }
.kpi-label { color: #91a2bb; font-size: .82rem; font-weight: 600; }
.kpi-value { color: #f3f7ff; font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; margin-top: .35rem; }
.kpi-note { color: #4be1a7; font-size: .76rem; margin-top: .2rem; }
.section-title { font-size: 1.25rem; font-weight: 700; margin: .4rem 0 .75rem 0; }
.small-muted { color: #91a2bb; font-size: .85rem; }
div[data-testid="stMetric"] { background: #141e31; border: 1px solid #27364e; padding: 14px 16px; border-radius: 15px; }
div[data-testid="stMetricLabel"] { color: #9aaac0; }
div[data-testid="stMetricValue"] { color: #f3f7ff; }
.stButton > button { background: #43dfa5; color: #07151a; border: 0; border-radius: 10px; font-weight: 700; padding: .65rem 1rem; }
.stButton > button:hover { background: #65e9b8; color: #07151a; }
.stDownloadButton > button { border-radius: 10px; }
hr { border-color: #26364d; }
[data-testid="stDataFrame"] { border: 1px solid #26364d; border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

DATA_CANDIDATES = [
    Path("ev_charging_volume_cleaned.csv"),
    Path("data/ev_charging_volume_cleaned.csv"),
    Path("ev_charging_volume_preview.csv"),
]
MODEL_CANDIDATES = [
    Path("ev_charging_pipeline.pkl"),
    Path("ev_charging_model.pkl"),
    Path("EV_Usage.pkl"),
]

@st.cache_data(show_spinner="Loading EV charging data...")
def load_data(path_string: str) -> pd.DataFrame:
    frame = pd.read_csv(path_string)
    frame.columns = [str(c).strip() for c in frame.columns]
    if "timestamp" not in frame.columns and "time" in frame.columns:
        frame = frame.rename(columns={"time": "timestamp"})
    required = {"timestamp", "station_id", "charging_volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["station_id"] = frame["station_id"].astype(str)
    frame["charging_volume"] = pd.to_numeric(frame["charging_volume"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "station_id", "charging_volume"]).copy()
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day"] = frame["timestamp"].dt.day
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    frame["month"] = frame["timestamp"].dt.month
    return frame

@st.cache_resource(show_spinner=False)
def load_model(path_string: str):
    return joblib.load(path_string)

def find_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None

def fmt_number(value, digits=1):
    return f"{value:,.{digits}f}"

def chart_layout(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce6f5", family="DM Sans"),
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#2b3b54", tickfont=dict(color="#9fb0c7"))
    fig.update_yaxes(gridcolor="#223149", zerolinecolor="#2b3b54", tickfont=dict(color="#9fb0c7"))
    return fig

# ----------------------------- Sidebar -----------------------------
st.sidebar.markdown("<div style='font-family:Space Grotesk;font-size:1.55rem;font-weight:700;'>⚡ Volt<span style='color:#43dfa5'>IQ</span></div>", unsafe_allow_html=True)
st.sidebar.caption("EV CHARGING ANALYTICS PLATFORM")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Station Explorer", "Demand Prediction", "Data Explorer"],
    label_visibility="visible",
)
st.sidebar.markdown("---")
data_path = find_existing(DATA_CANDIDATES)
if data_path is None:
    st.error("Dataset not found. Put `ev_charging_volume_cleaned.csv` in the same folder as `app.py`.")
    st.code("python -m pip install streamlit pandas plotly scikit-learn joblib\nstreamlit run app.py", language="bash")
    st.stop()

try:
    df = load_data(str(data_path))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

model_path = find_existing(MODEL_CANDIDATES)
model = None
if model_path is not None:
    try:
        model = load_model(str(model_path))
    except Exception:
        model = None

st.sidebar.markdown("**Dataset status**")
st.sidebar.markdown(f"<span style='color:#43dfa5'>●</span> Data loaded", unsafe_allow_html=True)
st.sidebar.caption(f"File: {data_path.name}")
st.sidebar.caption(f"{len(df):,} valid records · {df['station_id'].nunique():,} stations")
st.sidebar.markdown("**Model status**")
if model is not None:
    st.sidebar.markdown("<span style='color:#43dfa5'>●</span> Model ready", unsafe_allow_html=True)
    st.sidebar.caption(f"File: {model_path.name}")
else:
    st.sidebar.markdown("<span style='color:#f6bd60'>●</span> Model file not found", unsafe_allow_html=True)
    st.sidebar.caption("Analytics still work. Add ev_charging_pipeline.pkl to enable predictions.")

# ----------------------------- Header -----------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">SMART ENERGY · LIVE ANALYTICS</div>
  <h1>EV Charging Intelligence</h1>
  <p>Explore charging patterns, compare station performance, and estimate charging volume with a clean, data-driven workspace.</p>
</div>
""", unsafe_allow_html=True)

# Filters are shared by analytics pages
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()
with st.sidebar.expander("FILTER DATA", expanded=False):
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    station_options = sorted(df["station_id"].unique().tolist())
    chosen_stations = st.multiselect("Stations", station_options, default=[] , help="Leave empty to include all stations")

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date
filtered = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)]
if chosen_stations:
    filtered = filtered[filtered["station_id"].isin(chosen_stations)]
if filtered.empty:
    st.warning("No records match these filters. Widen the date range or clear the station filter.")
    st.stop()

# ----------------------------- Overview -----------------------------
if page == "Overview":
    st.markdown("<div class='section-title'>Network at a glance</div>", unsafe_allow_html=True)
    total_volume = filtered["charging_volume"].sum()
    avg_volume = filtered["charging_volume"].mean()
    active_stations = filtered["station_id"].nunique()
    peak_hour = int(filtered.groupby("hour")["charging_volume"].mean().idxmax())
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, note in [
        (k1, "TOTAL CHARGING VOLUME", fmt_number(total_volume), "Across selected records"),
        (k2, "AVERAGE VOLUME", fmt_number(avg_volume), "Per record"),
        (k3, "ACTIVE STATIONS", f"{active_stations:,}", "In selected period"),
        (k4, "PEAK AVERAGE HOUR", f"{peak_hour:02d}:00", "By average hourly volume"),
    ]:
        with col:
            st.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-note'>↗ {note}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("<div class='section-title'>Charging volume over time</div>", unsafe_allow_html=True)
        daily = filtered.set_index("timestamp").resample("D")["charging_volume"].sum().reset_index()
        fig = px.area(daily, x="timestamp", y="charging_volume", labels={"timestamp":"Date", "charging_volume":"Total volume"})
        fig.update_traces(line_color="#43dfa5", fillcolor="rgba(67,223,165,0.14)")
        st.plotly_chart(chart_layout(fig, 370), use_container_width=True)
    with right:
        st.markdown("<div class='section-title'>Average volume by hour</div>", unsafe_allow_html=True)
        hourly = filtered.groupby("hour", as_index=False)["charging_volume"].mean()
        fig = px.bar(hourly, x="hour", y="charging_volume", labels={"hour":"Hour of day", "charging_volume":"Average volume"})
        fig.update_traces(marker_color="#43dfa5", marker_line_width=0)
        st.plotly_chart(chart_layout(fig, 370), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("<div class='section-title'>Top stations by total volume</div>", unsafe_allow_html=True)
        top = filtered.groupby("station_id", as_index=False)["charging_volume"].sum().nlargest(10, "charging_volume").sort_values("charging_volume")
        fig = px.bar(top, x="charging_volume", y="station_id", orientation="h", labels={"charging_volume":"Total volume", "station_id":"Station"})
        fig.update_traces(marker_color="#6d91ff")
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)
    with right:
        st.markdown("<div class='section-title'>Volume distribution</div>", unsafe_allow_html=True)
        plot_values = filtered["charging_volume"]
        if len(plot_values) > 50000:
            plot_values = plot_values.sample(50000, random_state=42)
        fig = px.histogram(x=plot_values, nbins=45, labels={"x":"Charging volume", "count":"Records"})
        fig.update_traces(marker_color="#a78bfa", marker_line_width=0)
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)

    st.markdown("<div class='section-title'>Recent records</div>", unsafe_allow_html=True)
    st.dataframe(filtered[["timestamp", "station_id", "charging_volume"]].sort_values("timestamp", ascending=False).head(10), use_container_width=True, hide_index=True)

# ----------------------------- Station Explorer -----------------------------
elif page == "Station Explorer":
    st.markdown("<div class='section-title'>Station performance explorer</div>", unsafe_allow_html=True)
    station_summary = filtered.groupby("station_id").agg(
        records=("charging_volume", "size"),
        total_volume=("charging_volume", "sum"),
        average_volume=("charging_volume", "mean"),
        max_volume=("charging_volume", "max"),
    ).reset_index()
    station_summary = station_summary.sort_values("total_volume", ascending=False)
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_station = st.selectbox("Choose a station", sorted(filtered["station_id"].unique().tolist()))
        sdata = filtered[filtered["station_id"] == selected_station].sort_values("timestamp")
        st.metric("Total volume", fmt_number(sdata["charging_volume"].sum()))
        st.metric("Average volume", fmt_number(sdata["charging_volume"].mean()))
        st.metric("Records", f"{len(sdata):,}")
    with c2:
        fig = px.line(sdata, x="timestamp", y="charging_volume", labels={"timestamp":"Date and time", "charging_volume":"Charging volume"})
        fig.update_traces(line_color="#43dfa5", line_width=2)
        st.plotly_chart(chart_layout(fig, 390), use_container_width=True)
    st.markdown("<div class='section-title'>Station leaderboard</div>", unsafe_allow_html=True)
    fig = px.scatter(station_summary, x="average_volume", y="total_volume", size="records", color="average_volume", hover_name="station_id", labels={"average_volume":"Average volume", "total_volume":"Total volume", "records":"Records"}, color_continuous_scale="Mint")
    st.plotly_chart(chart_layout(fig, 420), use_container_width=True)
    st.dataframe(station_summary, use_container_width=True, hide_index=True)

# ----------------------------- Demand Prediction -----------------------------
elif page == "Demand Prediction":
    st.markdown("<div class='section-title'>Charging volume prediction</div>", unsafe_allow_html=True)
    st.markdown("Choose a station and time details to estimate its charging volume. Predictions require the trained model pipeline from your notebook.")
    if model is None:
        st.warning("The trained model was not found. Put `ev_charging_pipeline.pkl` in the same folder as `app.py`. If your notebook saved a different model file, rename it or update MODEL_CANDIDATES near the top of this script.")
        st.code("# In your notebook, download the saved model file\nfrom google.colab import files\nfiles.download('ev_charging_pipeline.pkl')", language="python")
    else:
        p1, p2 = st.columns(2)
        with p1:
            prediction_station = st.selectbox("Station ID", sorted(df["station_id"].unique().tolist()), key="pred_station")
            prediction_date = st.date_input("Prediction date", value=max_date, key="pred_date")
        with p2:
            prediction_hour = st.slider("Hour of day", min_value=0, max_value=23, value=17, format="%d:00")
            st.caption("Day of week and month are calculated automatically from the date.")
        if st.button("⚡ Predict charging volume", use_container_width=False):
            input_frame = pd.DataFrame([{
                "station_id": prediction_station,
                "hour": int(prediction_hour),
                "day": int(prediction_date.day),
                "day_of_week": int(prediction_date.weekday()),
                "month": int(prediction_date.month),
            }])
            try:
                prediction = float(np.asarray(model.predict(input_frame)).reshape(-1)[0])
                st.markdown("---")
                st.markdown("<div class='eyebrow'>MODEL ESTIMATE</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-family:Space Grotesk;font-size:3rem;font-weight:700;color:#43dfa5'>{prediction:,.2f}</div>", unsafe_allow_html=True)
                st.caption(f"Estimated charging volume for station {prediction_station} at {prediction_hour:02d}:00 on {prediction_date.strftime('%d %b %Y')}.")
                if prediction < 0:
                    st.info("The model returned a negative estimate. Treat it as a model limitation; the estimate is not clipped or presented as a measured value.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info("Check that the saved model is the complete pipeline trained with station_id, hour, day, day_of_week, and month features.")
    st.markdown("---")
    st.markdown("**Note:** This is a model-based estimate from historical patterns, not a live reading from a charging station.")

# ----------------------------- Data Explorer -----------------------------
else:
    st.markdown("<div class='section-title'>Explore and export records</div>", unsafe_allow_html=True)
    q1, q2, q3 = st.columns([1.2, 1, 1])
    with q1:
        search_station = st.text_input("Search station ID", placeholder="Type a station ID")
    with q2:
        min_volume = float(filtered["charging_volume"].min())
        max_volume = float(filtered["charging_volume"].max())
        lower = st.number_input("Minimum volume", value=min_volume, min_value=min_volume, max_value=max_volume)
    with q3:
        upper = st.number_input("Maximum volume", value=max_volume, min_value=min_volume, max_value=max_volume)
    table = filtered[(filtered["charging_volume"] >= lower) & (filtered["charging_volume"] <= upper)]
    if search_station.strip():
        table = table[table["station_id"].str.contains(search_station.strip(), case=False, na=False)]
    st.caption(f"Showing {len(table):,} of {len(filtered):,} filtered records")
    st.dataframe(table[["timestamp", "station_id", "charging_volume"]].sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True, height=430)
    csv_bytes = table[["timestamp", "station_id", "charging_volume"]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download filtered CSV", data=csv_bytes, file_name="ev_charging_filtered_data.csv", mime="text/csv")

st.markdown("---")
st.markdown("<div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#8192aa;font-size:.78rem'><span>⚡ VoltIQ · EV Charging Analytics</span><span>Built with Streamlit · Pandas · Plotly · Scikit-learn</span></div>", unsafe_allow_html=True)
