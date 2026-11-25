"""Shared code for the Tetouan day-ahead load forecast.
Used by the notebooks and by app.py, so training and serving build identical features."""
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CAL = ["hour_sin", "hour_cos", "is_weekend", "dow"]
REL = CAL + ["lag_24_rel", "lag_168_rel", "yday_mean_rel", "temp", "cdd20"]
REL_NOWX = [c for c in REL if c not in ("temp", "cdd20")]   # final model: no weather input

MODELS = {
    "linear": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "rf": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                        n_jobs=-1, random_state=0),
    "lgbm": lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                      random_state=0, verbose=-1),
}

def make_tscv():
    return TimeSeriesSplit(n_splits=6, test_size=24 * 28)

def build_features(h):
    """h: hourly frame with 'total' and 'Temperature'. Forecast origin = midnight,
    so every demand feature uses data at least 24 h old."""
    y = h["total"]
    hr, dow = y.index.hour, np.asarray(y.index.day_of_week)
    F = pd.DataFrame({"actual": y})
    F["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    F["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    F["is_weekend"] = (dow >= 5).astype(int)
    F["dow"] = dow
    F["lag_24"] = y.shift(24)
    F["lag_168"] = y.shift(168)
    F["daytype_lag"] = np.where(np.isin(dow, [1, 2, 3, 4]), F["lag_24"], F["lag_168"])
    F["roll_mean_7d"] = y.shift(24).rolling(168).mean()
    daily = y.resample("D").mean()
    F["yday_mean"] = daily.shift(1).reindex(y.index.normalize()).to_numpy()
    F["temp"] = h["Temperature"]
    F["cdd20"] = np.maximum(F["temp"] - 20, 0)
    for c in ["lag_24", "lag_168", "yday_mean"]:
        F[c + "_rel"] = F[c] / F["roll_mean_7d"]
    return F

def cv_predict_rel(F, tscv, make_model, cols):
    out = []
    for i, (tr, te) in enumerate(tscv.split(F), start=1):
        Xtr = F.iloc[tr][cols]
        ytr = F.iloc[tr]["actual"] / F.iloc[tr]["roll_mean_7d"]
        ok = Xtr.notna().all(axis=1) & ytr.notna()
        m = make_model().fit(Xtr[ok], ytr[ok])
        p = m.predict(F.iloc[te][cols]) * F.iloc[te]["roll_mean_7d"].to_numpy()
        out.append(pd.DataFrame({"fold": i, "pred": p}, index=F.index[te]))
    return pd.concat(out)

def oof_from_col(F, tscv, col):
    return pd.concat([pd.DataFrame({"fold": i, "pred": F.iloc[te][col].to_numpy()},
                                   index=F.index[te])
                      for i, (_, te) in enumerate(tscv.split(F), start=1)])

def train_final(F):
    X = F[REL_NOWX]
    y = F["actual"] / F["roll_mean_7d"]
    ok = X.notna().all(axis=1) & y.notna()
    return MODELS["lgbm"]().fit(X[ok], y[ok])

def predict_next_day(model, history):
    """history: hourly Series of total load ending at 23:00 of the day before the
    target day, at least 192 hours. Returns 24 hourly forecasts for the next day."""
    history = history.sort_index()
    if len(history) < 192:
        raise ValueError("Need at least 192 hours (8 days) of history")
    if history.index[-1].hour != 23:
        raise ValueError("History must end at 23:00")
    if not (history.index.to_series().diff().dropna() == pd.Timedelta("1h")).all():
        raise ValueError("History has gaps or duplicates")
    future = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    y_ext = pd.concat([history, pd.Series(np.nan, index=future)])
    Fx = build_features(pd.DataFrame({"total": y_ext, "Temperature": np.nan}))
    X = Fx.loc[future, REL_NOWX]
    if X.isna().any().any():
        raise ValueError("Not enough history for the features")
    pred = model.predict(X) * Fx.loc[future, "roll_mean_7d"].to_numpy()
    return pd.Series(pred, index=future, name="forecast")


def day_type_baseline(history):
    """Naive baseline for the next day, same rule as the 'daytype_lag' feature:
    Tue-Fri copy yesterday, Mon/Sat/Sun copy the same day last week.
    history: hourly Series of total load ending at 23:00 (at least 168 hours)."""
    history = history.sort_index()
    if len(history) < 168:
        raise ValueError("Need at least 168 hours of history")
    day = history.index[-1].normalize() + pd.Timedelta(days=1)
    vals = history.iloc[-24:] if day.day_of_week in (1, 2, 3, 4) else history.iloc[-168:-144]
    return pd.Series(vals.to_numpy(), index=pd.date_range(day, periods=24, freq="h"), name="baseline")
