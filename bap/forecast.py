"""Lightweight forecasting: linear trend + seasonal dummies fitted by least squares.

Chosen over heavy deep-learning models on purpose: it trains in milliseconds on a
CPU, is fully explainable to finance users, and gives prediction intervals from
the residual distribution. A holdout backtest reports MAPE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _design(t: np.ndarray, season_idx: np.ndarray, period: int) -> np.ndarray:
    cols = [np.ones_like(t, dtype=float), t.astype(float)]
    for s in range(1, period):
        cols.append((season_idx == s).astype(float))
    return np.column_stack(cols)


def fit_forecast(series: pd.Series, horizon: int, period: int = 12, z: float = 1.64) -> pd.DataFrame:
    """series: indexed by Timestamp at regular frequency (monthly or quarterly)."""
    s = series.dropna().astype(float)
    n = len(s)
    freq = "MS" if period == 12 else "QS"
    season_of = (lambda idx: idx.month - 1) if period == 12 else (lambda idx: idx.quarter - 1)
    t = np.arange(n)
    X = _design(t, np.asarray(season_of(s.index)), period) if n >= period + 3 else _design(t, np.zeros(n, int), 1)
    beta, *_ = np.linalg.lstsq(X, s.to_numpy(), rcond=None)
    fitted = X @ beta
    resid_sd = float(np.std(s.to_numpy() - fitted, ddof=min(X.shape[1], n - 1)))

    fut_idx = pd.date_range(s.index[-1], periods=horizon + 1, freq=freq)[1:]
    tf = np.arange(n, n + horizon)
    Xf = _design(tf, np.asarray(season_of(fut_idx)), period) if X.shape[1] > 2 else _design(tf, np.zeros(horizon, int), 1)
    yhat = Xf @ beta
    widen = np.sqrt(1 + np.arange(1, horizon + 1) / n)
    out = pd.DataFrame({"actual": s, "fitted": fitted}, index=s.index)
    fut = pd.DataFrame({"forecast": yhat, "lower": yhat - z * resid_sd * widen, "upper": yhat + z * resid_sd * widen}, index=fut_idx)
    return pd.concat([out, fut])


def backtest_mape(series: pd.Series, holdout: int, period: int = 12) -> float:
    s = series.dropna()
    if len(s) <= holdout + 4:
        return float("nan")
    fc = fit_forecast(s.iloc[:-holdout], holdout, period)["forecast"].dropna()
    actual = s.iloc[-holdout:].to_numpy()
    return float(np.mean(np.abs((actual - fc.to_numpy()[:holdout]) / actual)) * 100)


def ledger_monthly(df: pd.DataFrame, account_type: str = "Revenue", category: str | None = None,
                   business_unit: str | None = None) -> pd.Series:
    d = df[df["account_type"] == account_type]
    if category:
        d = d[d["category"] == category]
    if business_unit:
        d = d[d["business_unit"] == business_unit]
    if "is_injected_anomaly" in d:  # forecast on cleaned data
        d = d[d["is_injected_anomaly"] == 0]
    return d.groupby("month")["amount_eur"].sum().asfreq("MS").fillna(0)
