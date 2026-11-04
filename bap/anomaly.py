"""Hybrid anomaly detection on the general ledger.

1. Explainable control rules that auditors already use (duplicates, split
   payments below the approval threshold, off-hours postings, unknown vendors,
   extreme amounts vs. category norm).
2. An unsupervised Isolation Forest over engineered features to catch unusual
   combinations the rules do not encode.
The final risk score blends both; every flag carries human-readable reasons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

APPROVAL_THRESHOLD = 10_000


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    e = df[df["account_type"] == "Expense"].copy()
    e["log_amount"] = np.log1p(e["amount_eur"])
    g = e.groupby("category")["log_amount"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median()).replace(0, 1e-6)
    e["robust_z"] = 0.6745 * (e["log_amount"] - med) / mad
    e["hour"] = e["txn_date"].dt.hour
    e["is_weekend"] = (e["txn_date"].dt.dayofweek >= 5).astype(int)
    e["off_hours"] = ((e["hour"] < 6) | (e["hour"] >= 22) | (e["is_weekend"] == 1)).astype(int)
    e["is_round"] = ((e["amount_eur"] % 1000) == 0).astype(int)
    vf = e["counterparty"].map(e["counterparty"].value_counts())
    e["vendor_freq"] = vf
    e["log_vendor_freq"] = np.log1p(vf)
    e["near_threshold"] = e["amount_eur"].between(APPROVAL_THRESHOLD * 0.95, APPROVAL_THRESHOLD - 0.01).astype(int)

    # duplicates: same vendor + same amount within 60 minutes (flag the later one)
    e = e.sort_values("txn_date")
    prev_t = e.groupby(["counterparty", "amount_eur"])["txn_date"].shift(1)
    e["is_duplicate"] = ((e["txn_date"] - prev_t) <= pd.Timedelta(minutes=60)).astype(int)

    # split payments: >=3 near-threshold payments to the same vendor within 24h
    nt = e[e["near_threshold"] == 1]
    split_ids: set = set()
    for _, grp in nt.groupby("counterparty"):
        t = grp["txn_date"].to_numpy()
        ids = grp["txn_id"].to_numpy()
        for i in range(len(t)):
            win = (t >= t[i] - np.timedelta64(24, "h")) & (t <= t[i] + np.timedelta64(24, "h"))
            if win.sum() >= 3:
                split_ids.add(ids[i])
    e["is_split"] = e["txn_id"].isin(split_ids).astype(int)
    return e


FEATURES = ["log_amount", "robust_z", "hour", "is_weekend", "is_round", "log_vendor_freq", "near_threshold"]


def detect(df: pd.DataFrame, contamination: float = 0.004, seed: int = 7) -> pd.DataFrame:
    e = build_features(df)
    iso = IsolationForest(n_estimators=150, contamination=contamination, random_state=seed, n_jobs=1)
    iso.fit(e[FEATURES])
    raw = -iso.score_samples(e[FEATURES])
    e["ml_score"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    e["ml_flag"] = (iso.predict(e[FEATURES]) == -1).astype(int)

    rules = {
        "Duplicate payment (same vendor & amount within 60 min)": e["is_duplicate"] == 1,
        "Possible split payment below EUR 10k dual-approval limit": e["is_split"] == 1,
        "Large round amount posted off-hours/weekend": (e["off_hours"] == 1) & (e["amount_eur"] >= 20_000),
        "Rarely used vendor with large payment": (e["vendor_freq"] < 30) & (e["amount_eur"] >= 20_000),
        "Amount far above category norm": e["robust_z"] > 6,
    }
    reasons = pd.Series([[] for _ in range(len(e))], index=e.index)
    rule_hits = np.zeros(len(e))
    for label, mask in rules.items():
        for i in e.index[mask]:
            reasons[i].append(label)
        rule_hits += mask.to_numpy()
    e["rule_hits"] = rule_hits
    e["reasons"] = reasons.apply(lambda r: "; ".join(r))
    e["risk_score"] = np.clip(0.55 * np.minimum(rule_hits, 2) / 2 + 0.45 * e["ml_score"], 0, 1).round(3)
    e["flagged"] = ((e["rule_hits"] > 0) | (e["ml_flag"] == 1)).astype(int)
    e.loc[(e["flagged"] == 1) & (e["reasons"] == ""), "reasons"] = "Unusual feature combination (Isolation Forest)"
    return e.sort_values("risk_score", ascending=False)


def evaluate(scored: pd.DataFrame) -> dict:
    if "is_injected_anomaly" not in scored:
        return {}
    y, p = scored["is_injected_anomaly"], scored["flagged"]
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    by_type = (scored[scored["is_injected_anomaly"] == 1].groupby("anomaly_type")["flagged"].mean() * 100).round(1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "recall_by_type": by_type.to_dict(), "flagged": int(p.sum()), "reviewed_pool": len(scored)}
