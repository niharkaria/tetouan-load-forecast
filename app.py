import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
import forecast_lib as fl  # same code as training: no re-implemented features here

st.set_page_config(page_title="Tetouan day-ahead load forecast", layout="wide")
st.title("Tetouan day-ahead load forecast")
st.caption("LightGBM on load ratios, no weather input. The forecast is made at midnight "
           "and the model only ever sees past load.")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "lgbm_ratio_noweather.joblib")


@st.cache_data
def load_demo():
    return pd.read_csv(ROOT / "data" / "demo" / "demo_history.csv",
                       index_col=0, parse_dates=True)["total"]


bundle = load_model()
model = bundle["model"]

mode = st.radio("Mode", ["Backtest demo (Dec 3 to 30)", "Upload my own history"],
                horizontal=True)

if mode.startswith("Backtest"):
    demo = load_demo()
    first = pd.Timestamp(bundle["trained_until"]).normalize() + pd.Timedelta(days=1)
    last = demo.index[-1].normalize()
    day = st.date_input("Day to forecast", value=first.date(),
                        min_value=first.date(), max_value=last.date())
    day = pd.Timestamp(day)

    history = demo.loc[: day - pd.Timedelta(hours=1)]
    try:
        forecast = fl.predict_next_day(model, history)
        baseline = fl.day_type_baseline(history)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    actual = demo.loc[day: day + pd.Timedelta(hours=23)]
    c1, c2 = st.columns(2)
    c1.metric("Model MAE for this day", f"{(forecast - actual).abs().mean():,.0f}")
    c2.metric("Day-type baseline MAE", f"{(baseline - actual).abs().mean():,.0f}")
    st.line_chart(pd.DataFrame({"actual": actual, "model": forecast, "baseline": baseline}))
    st.caption(f"The model's weights were trained on data up to {bundle['trained_until']}, so these "
               "days were not in training. They are also walk-forward fold 6, which was used when "
               "choosing the model, so this is a demo, not an independent test. One day is a noisy "
               "sample: the six-fold results in the README are the evidence.")
else:
    st.write("Upload a CSV whose first column is the timestamp and which has a `total` load "
             "column. Hourly, no gaps, ending at 23:00, at least 192 rows (8 days). "
             "The model was trained on Tetouan load, so data on another scale will not work.")
    uploaded = st.file_uploader("History CSV", type="csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, index_col=0, parse_dates=True)
            forecast = fl.predict_next_day(model, df["total"])
        except (ValueError, KeyError) as e:
            st.error(f"Could not forecast: {e}")
            st.stop()
        st.line_chart(forecast)
        st.dataframe(forecast.round(0).to_frame())

st.info("Limits: one year of data, so yearly seasonality and holidays such as Eid cannot be "
        "learned or validated. Weather is not used. The model's gain over a naive baseline "
        "is modest and not statistically clear outside Eid week.")
