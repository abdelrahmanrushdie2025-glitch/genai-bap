"""SEC EDGAR XBRL ingestion (free public API, no key required).

Pulls 'companyfacts' JSON for a peer group of public US financial institutions
and normalises quarterly revenue and net income. SEC asks for a descriptive
User-Agent with a contact address and max ~10 requests/second.
Docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""
from __future__ import annotations

import os
import re
import time

import pandas as pd
import requests

from .data import SEC_LIVE

PEERS = {  # ticker: (company, CIK)
    "JPM": ("JPMorgan Chase", 19617),
    "BAC": ("Bank of America", 70858),
    "C": ("Citigroup", 831001),
    "WFC": ("Wells Fargo", 72971),
    "GS": ("Goldman Sachs", 886982),
    "MS": ("Morgan Stanley", 895421),
    "AXP": ("American Express", 4962),
    "V": ("Visa", 1403161),
    "MA": ("Mastercard", 1141391),
}
REVENUE_CONCEPTS = ["RevenuesNetOfInterestExpense", "Revenues",
                    "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
NET_INCOME_CONCEPTS = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        try:
            import streamlit as st
            ua = st.secrets.get("SEC_USER_AGENT")
        except Exception:
            ua = None
    return ua or "GenAI-BAP university prototype (student project; set SEC_USER_AGENT)"


def quarterly_series(facts: dict, concepts: list[str]) -> pd.Series:
    """Return a quarterly USD series keyed by calendar quarter 'YYYYQn'.

    Uses XBRL 'frame' values: CYyyyyQn = single-quarter duration. When Q4 is
    only reported inside the annual 10-K (frame CYyyyy), derive Q4 = FY - Q1..Q3.
    The first concept with the most data wins.
    """
    best = pd.Series(dtype=float)
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        units = gaap.get(concept, {}).get("units", {}).get("USD", [])
        q, fy = {}, {}
        for u in units:
            frame = u.get("frame", "")
            if re.fullmatch(r"CY\d{4}Q[1-4]", frame):
                q[frame[2:]] = float(u["val"])
            elif re.fullmatch(r"CY\d{4}", frame):
                fy[frame[2:]] = float(u["val"])
        for year, total in fy.items():
            k4 = f"{year}Q4"
            parts = [q.get(f"{year}Q{i}") for i in (1, 2, 3)]
            if k4 not in q and all(p is not None for p in parts):
                q[k4] = total - sum(parts)
        s = pd.Series(q, dtype=float).sort_index()
        if len(s) > len(best):
            best = s
    return best


def _quarter_end(fq: str) -> pd.Timestamp:
    return pd.Period(fq, freq="Q").end_time.normalize()


def normalise(ticker: str, company: str, facts: dict, since: str = "2019Q1") -> pd.DataFrame:
    rev = quarterly_series(facts, REVENUE_CONCEPTS)
    ni = quarterly_series(facts, NET_INCOME_CONCEPTS)
    df = pd.DataFrame({"revenue_usd": rev, "net_income_usd": ni}).dropna(subset=["revenue_usd"])
    df = df[df.index >= since]
    df.index.name = "fiscal_quarter"
    df = df.reset_index()
    df.insert(0, "company", company)
    df.insert(0, "ticker", ticker)
    df["period_end"] = df["fiscal_quarter"].map(_quarter_end)
    return df[["ticker", "company", "period_end", "fiscal_quarter", "revenue_usd", "net_income_usd"]]


def fetch_all(session: requests.Session | None = None, peers: dict = PEERS, save: bool = True) -> tuple[pd.DataFrame, list[str]]:
    s = session or requests.Session()
    s.headers.update({"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"})
    frames, errors = [], []
    for ticker, (name, cik) in peers.items():
        try:
            r = s.get(URL.format(cik=cik), timeout=30)
            r.raise_for_status()
            frames.append(normalise(ticker, name, r.json()))
        except Exception as e:  # keep going - partial data is still useful
            errors.append(f"{ticker}: {e}")
        time.sleep(0.15)  # stay well under SEC's 10 req/s fair-access limit
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if save and not out.empty:
        SEC_LIVE.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(SEC_LIVE, index=False)
    return out, errors
