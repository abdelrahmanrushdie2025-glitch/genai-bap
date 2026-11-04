"""Generate the synthetic enterprise ledger for the fictional company
"Aurora Meridian Group" (AMG).

Real corporate ledgers are confidential, so - like the widely used PaySim
dataset on Kaggle - we simulate one. Known anomalies are injected and labelled
so the anomaly detector can be evaluated (precision / recall).

Run:  python scripts/generate_data.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

SEED = 42
START, END = "2023-01-01", "2026-09-30"
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ledger.csv.gz"

BUSINESS_UNITS = {
    # unit: (revenue weight, expense weight)
    "Retail Banking": (0.30, 0.22),
    "Corporate Banking": (0.28, 0.15),
    "Wealth Management": (0.22, 0.10),
    "Treasury": (0.12, 0.06),
    "Operations": (0.03, 0.17),
    "Technology": (0.02, 0.18),
    "Compliance & Risk": (0.01, 0.07),
    "Marketing": (0.02, 0.05),
}
REGIONS = ["Germany", "France", "Netherlands", "Spain", "Italy", "Nordics", "UK"]
REGION_W = [0.30, 0.18, 0.12, 0.10, 0.10, 0.08, 0.12]

REVENUE_CATS = {
    "Net Interest Income": (18000, 0.9),
    "Advisory Fees": (9000, 1.0),
    "Card & Payment Fees": (1200, 0.7),
    "Asset Management Fees": (7000, 0.8),
    "Trading Income": (12000, 1.2),
}
EXPENSE_CATS = {
    "Cloud & Software": (4200, 0.8, ["Azure Cloud Services", "Snowflake EMEA", "Atlassian", "ServiceNow", "Datadog EU"]),
    "Salaries & Benefits": (6500, 0.5, ["Payroll Run"]),
    "Professional Services": (8000, 0.9, ["Kessler & Partner Consulting", "Northbridge Legal", "Veritas Audit GmbH"]),
    "Office & Facilities": (2500, 0.7, ["Lumen Facility Mgmt", "CityPower Energy", "Office Depot EU"]),
    "Marketing & Media": (3500, 0.9, ["Brightline Media", "Google Ads EMEA", "LinkedIn Ireland"]),
    "Data & Market Feeds": (5200, 0.6, ["Bloomberg L.P.", "Refinitiv", "S&P Global"]),
    "Travel": (900, 0.8, ["Lufthansa", "Air France", "Booking Business"]),
    "Regulatory Fees": (6000, 0.5, ["BaFin", "ECB Supervisory Levy", "AMF France"]),
}
PAYMENT_METHODS = ["SEPA Transfer", "Corporate Card", "Direct Debit", "Internal Allocation"]
APPROVAL_THRESHOLD = 10_000  # EUR - payments above need dual approval


def _dates(rng: np.random.Generator, n: int) -> pd.DatetimeIndex:
    days = pd.date_range(START, END, freq="D")
    # business days carry most of the volume; mild growth + Q4 seasonality
    w = np.where(days.dayofweek < 5, 1.0, 0.08)
    t = np.arange(len(days)) / len(days)
    w = w * (1 + 0.35 * t) * (1 + 0.12 * np.isin(days.month, [10, 11, 12]))
    idx = rng.choice(len(days), size=n, p=w / w.sum())
    hours = rng.normal(13, 2.3, n).clip(7, 19)
    minutes = rng.integers(0, 60, n)
    return pd.DatetimeIndex(days[idx]) + pd.to_timedelta(hours.astype(int), "h") + pd.to_timedelta(minutes, "m")


def generate(n_revenue: int = 26_000, n_expense: int = 34_000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    units = list(BUSINESS_UNITS)

    # ---------- revenue ----------
    rw = np.array([BUSINESS_UNITS[u][0] for u in units]); rw /= rw.sum()
    rcats = list(REVENUE_CATS)
    rev = pd.DataFrame({
        "txn_date": _dates(rng, n_revenue),
        "business_unit": rng.choice(units, n_revenue, p=rw),
        "account_type": "Revenue",
        "category": rng.choice(rcats, n_revenue, p=[0.34, 0.16, 0.22, 0.16, 0.12]),
    })
    mu = rev["category"].map(lambda c: REVENUE_CATS[c][0]).to_numpy()
    sd = rev["category"].map(lambda c: REVENUE_CATS[c][1]).to_numpy()
    rev["amount_eur"] = np.round(rng.lognormal(np.log(mu), sd * 0.55), 2)
    rev["counterparty"] = rng.choice([f"Client-{i:04d}" for i in range(1, 1800)], n_revenue)
    rev["payment_method"] = rng.choice(["SEPA Transfer", "Internal Allocation"], n_revenue, p=[0.8, 0.2])

    # ---------- expenses ----------
    ew = np.array([BUSINESS_UNITS[u][1] for u in units]); ew /= ew.sum()
    ecats = list(EXPENSE_CATS)
    exp = pd.DataFrame({
        "txn_date": _dates(rng, n_expense),
        "business_unit": rng.choice(units, n_expense, p=ew),
        "account_type": "Expense",
        "category": rng.choice(ecats, n_expense, p=[0.16, 0.20, 0.12, 0.12, 0.10, 0.10, 0.14, 0.06]),
    })
    mu = exp["category"].map(lambda c: EXPENSE_CATS[c][0]).to_numpy()
    sd = exp["category"].map(lambda c: EXPENSE_CATS[c][1]).to_numpy()
    exp["amount_eur"] = np.round(rng.lognormal(np.log(mu), sd * 0.5), 2)
    exp["counterparty"] = [rng.choice(EXPENSE_CATS[c][2]) for c in exp["category"]]
    exp["payment_method"] = rng.choice(PAYMENT_METHODS, n_expense, p=[0.6, 0.15, 0.15, 0.10])
    # cloud spend grows faster (cloud migration story)
    cloud = exp["category"] == "Cloud & Software"
    growth = 1 + 0.6 * (exp["txn_date"] - pd.Timestamp(START)).dt.days / 1369
    exp.loc[cloud, "amount_eur"] = np.round(exp.loc[cloud, "amount_eur"] * growth[cloud], 2)

    df = pd.concat([rev, exp], ignore_index=True)
    df["region"] = rng.choice(REGIONS, len(df), p=REGION_W)
    df["is_injected_anomaly"] = 0
    df["anomaly_type"] = ""

    # ---------- inject labelled anomalies ----------
    anomalies = []
    exp_idx = df.index[df["account_type"] == "Expense"].to_numpy()

    # 1. extreme outlier amounts (fat-finger / fraud)
    for i in rng.choice(exp_idx, 60, replace=False):
        r = df.loc[i].copy()
        r["amount_eur"] = round(float(r["amount_eur"]) * rng.uniform(15, 60), 2)
        r["anomaly_type"] = "Extreme amount"
        anomalies.append(r)
    # 2. duplicate payments (same vendor, same amount, within minutes)
    for i in rng.choice(exp_idx, 70, replace=False):
        r = df.loc[i].copy()
        r["txn_date"] = r["txn_date"] + pd.Timedelta(minutes=int(rng.integers(1, 30)))
        r["anomaly_type"] = "Duplicate payment"
        anomalies.append(r)
    # 3. split payments just below the dual-approval threshold
    for _ in range(25):
        base = df.loc[rng.choice(exp_idx)].copy()
        for k in range(3):
            r = base.copy()
            r["amount_eur"] = round(rng.uniform(9_500, 9_990), 2)
            r["txn_date"] = base["txn_date"] + pd.Timedelta(hours=k * 2)
            r["category"] = "Professional Services"
            r["counterparty"] = "Kessler & Partner Consulting"
            r["anomaly_type"] = "Split below approval threshold"
            anomalies.append(r)
    # 4. off-hours postings to an unknown vendor with round amounts
    for _ in range(45):
        r = df.loc[rng.choice(exp_idx)].copy()
        d = r["txn_date"].normalize() + pd.Timedelta(days=int((5 - r["txn_date"].dayofweek) % 7))
        r["txn_date"] = d + pd.Timedelta(hours=int(rng.choice([1, 2, 3, 23])), minutes=int(rng.integers(0, 60)))
        r["amount_eur"] = float(rng.choice([25_000, 40_000, 50_000, 75_000]))
        r["counterparty"] = rng.choice(["Orion Trade Ltd", "Bluefin Holdings SA", "Nexa Global FZE"])
        r["payment_method"] = "SEPA Transfer"
        r["anomaly_type"] = "Off-hours unknown vendor"
        anomalies.append(r)

    an = pd.DataFrame(anomalies)
    an["is_injected_anomaly"] = 1
    df = pd.concat([df, an], ignore_index=True).sort_values("txn_date").reset_index(drop=True)
    df.insert(0, "txn_id", [f"AMG-{i:07d}" for i in range(1, len(df) + 1)])
    df["txn_date"] = df["txn_date"].dt.strftime("%Y-%m-%d %H:%M")
    df["approval_level"] = np.where(df["amount_eur"] >= APPROVAL_THRESHOLD, "Dual", "Single")
    return df[["txn_id", "txn_date", "business_unit", "account_type", "category", "counterparty",
               "region", "payment_method", "amount_eur", "approval_level", "is_injected_anomaly", "anomaly_type"]]


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {len(data):,} rows -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB); "
          f"anomalies={int(data.is_injected_anomaly.sum())}")


def generate_sec_demo(path: pathlib.Path = ROOT / "data" / "sec_demo.csv") -> pd.DataFrame:
    """Offline placeholder for the SEC peer table (clearly anonymised + synthetic).
    The app replaces it with real EDGAR data via the 'Refresh SEC data' button."""
    rng = np.random.default_rng(SEED + 1)
    peers = {"PEER-A": ("Demo Peer Bank A", 38e9), "PEER-B": ("Demo Peer Bank B", 24e9),
             "PEER-C": ("Demo Peer Bank C", 20e9), "PEER-D": ("Demo Payments Co D", 8e9),
             "PEER-E": ("Demo Investment Bank E", 12e9)}
    quarters = pd.period_range("2019Q1", "2026Q2", freq="Q")
    rows = []
    for t, (name, base) in peers.items():
        for i, q in enumerate(quarters):
            rev = base * (1 + 0.012 * i) * (1 + 0.03 * np.sin(i * 1.57)) * rng.normal(1, 0.03)
            margin = np.clip(rng.normal(0.27 if "Payments" not in name else 0.5, 0.04), 0.05, 0.7)
            rows.append({"ticker": t, "company": name, "period_end": q.end_time.normalize().date(),
                         "fiscal_quarter": str(q), "revenue_usd": round(rev, -6), "net_income_usd": round(rev * margin, -6)})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    generate_sec_demo()
    print("wrote data/sec_demo.csv")
