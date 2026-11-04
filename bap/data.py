"""Data access layer: loads the ledger, SEC peer data and internal documents."""
from __future__ import annotations

import pathlib
import sqlite3

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCS = ROOT / "docs"
LEDGER = DATA / "ledger.csv.gz"
SEC_LIVE = DATA / "sec_quarterly.csv"
SEC_DEMO = DATA / "sec_demo.csv"

# Columns the LLM is allowed to see. The ground-truth anomaly label is hidden
# from the query engine (governance: evaluation labels never leak into answers).
HIDDEN_COLUMNS = {"is_injected_anomaly", "anomaly_type"}


def load_ledger() -> pd.DataFrame:
    df = pd.read_csv(LEDGER, parse_dates=["txn_date"])
    df["month"] = df["txn_date"].dt.to_period("M").dt.to_timestamp()
    return df


def load_sec() -> tuple[pd.DataFrame, str]:
    """Return (quarterly peer financials, source label)."""
    if SEC_LIVE.exists():
        return pd.read_csv(SEC_LIVE, parse_dates=["period_end"]), "SEC EDGAR (live XBRL API)"
    return pd.read_csv(SEC_DEMO, parse_dates=["period_end"]), "Demo sample (synthetic) - run SEC refresh for live data"


def load_documents() -> list[dict]:
    docs = []
    for p in sorted(DOCS.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else p.stem
        docs.append({"id": p.stem, "title": title, "text": text})
    return docs


def sqlite_connection(ledger: pd.DataFrame, sec: pd.DataFrame | None = None) -> sqlite3.Connection:
    """In-memory, read-only SQLite warehouse used by the NL->SQL engine."""
    con = sqlite3.connect(":memory:", check_same_thread=False)
    led = ledger.drop(columns=[c for c in HIDDEN_COLUMNS if c in ledger], errors="ignore").copy()
    led["txn_date"] = led["txn_date"].dt.strftime("%Y-%m-%d %H:%M")
    led["month"] = pd.to_datetime(led["month"]).dt.strftime("%Y-%m")
    led.to_sql("ledger", con, index=False)
    if sec is not None:
        s = sec.copy()
        s["period_end"] = pd.to_datetime(s["period_end"]).dt.strftime("%Y-%m-%d")
        s.to_sql("sec_financials", con, index=False)
    con.execute("PRAGMA query_only = ON")
    return con


SCHEMA_DESCRIPTION = """
Table ledger  -- one row per general-ledger transaction of Aurora Meridian Group (EUR)
  txn_id TEXT, txn_date TEXT 'YYYY-MM-DD HH:MM', month TEXT 'YYYY-MM',
  business_unit TEXT  (Retail Banking, Corporate Banking, Wealth Management, Treasury,
                       Operations, Technology, Compliance & Risk, Marketing),
  account_type TEXT   ('Revenue' or 'Expense'),
  category TEXT       (Revenue: Net Interest Income, Advisory Fees, Card & Payment Fees,
                       Asset Management Fees, Trading Income;
                       Expense: Cloud & Software, Salaries & Benefits, Professional Services,
                       Office & Facilities, Marketing & Media, Data & Market Feeds, Travel,
                       Regulatory Fees),
  counterparty TEXT, region TEXT (Germany, France, Netherlands, Spain, Italy, Nordics, UK),
  payment_method TEXT, amount_eur REAL (always positive), approval_level TEXT ('Single','Dual')

Table sec_financials  -- quarterly results of public US financial peers (USD)
  ticker TEXT, company TEXT, period_end TEXT 'YYYY-MM-DD', fiscal_quarter TEXT 'YYYYQn',
  revenue_usd REAL, net_income_usd REAL
"""
