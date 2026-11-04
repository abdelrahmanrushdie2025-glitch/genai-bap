"""Natural-language querying: question -> safe SQL -> result -> plain-English summary."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from . import llm
from .data import SCHEMA_DESCRIPTION

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|detach|pragma|replace|vacuum|reindex)\b", re.I)

SQL_SYSTEM = (
    "You are a senior financial data analyst. Translate the user's question into ONE SQLite "
    "SELECT query over the schema below. Rules: read-only SELECT/WITH only; use exact column "
    "and category names; round money to 2 decimals; alias columns in snake_case; prefer "
    "aggregated results (<= 50 rows); for time series order by period ascending. "
    "Return ONLY the SQL inside a ```sql code block.\n" + SCHEMA_DESCRIPTION
)
SUMMARY_SYSTEM = (
    "You are a CFO's analyst. Given a question, the SQL used and the result table, write a "
    "concise executive answer (max 4 sentences) in English with the key numbers formatted "
    "(e.g. EUR 12.4M). Mention one notable insight. Do not invent numbers not in the table."
)


@dataclass
class QueryResult:
    question: str
    sql: str
    df: pd.DataFrame
    summary: str
    mode: str                     # "gemini", "gemini (cached)" or "offline"
    notes: list[str] = field(default_factory=list)


# ---------------- offline intent library (works without an API key) ----------------
OFFLINE_INTENTS = [
    (["revenue", "business unit"], "Revenue by business unit",
     "SELECT business_unit, ROUND(SUM(amount_eur),2) AS revenue_eur FROM ledger "
     "WHERE account_type='Revenue' GROUP BY 1 ORDER BY 2 DESC"),
    (["expense", "business unit"], "Expenses by business unit",
     "SELECT business_unit, ROUND(SUM(amount_eur),2) AS expenses_eur FROM ledger "
     "WHERE account_type='Expense' GROUP BY 1 ORDER BY 2 DESC"),
    (["profit", "year"], "Revenue, expenses and operating profit by year",
     "SELECT substr(month,1,4) AS year, "
     "ROUND(SUM(CASE WHEN account_type='Revenue' THEN amount_eur ELSE 0 END),2) AS revenue_eur, "
     "ROUND(SUM(CASE WHEN account_type='Expense' THEN amount_eur ELSE 0 END),2) AS expenses_eur, "
     "ROUND(SUM(CASE WHEN account_type='Revenue' THEN amount_eur ELSE -amount_eur END),2) AS operating_profit_eur "
     "FROM ledger GROUP BY 1 ORDER BY 1"),
    (["cloud", "month"], "Monthly cloud & software spend",
     "SELECT month, ROUND(SUM(amount_eur),2) AS cloud_spend_eur FROM ledger "
     "WHERE category='Cloud & Software' GROUP BY 1 ORDER BY 1"),
    (["top", "vendor"], "Top 10 vendors by spend",
     "SELECT counterparty AS vendor, COUNT(*) AS payments, ROUND(SUM(amount_eur),2) AS spend_eur "
     "FROM ledger WHERE account_type='Expense' GROUP BY 1 ORDER BY 3 DESC LIMIT 10"),
    (["expense", "category"], "Expenses by category",
     "SELECT category, ROUND(SUM(amount_eur),2) AS expenses_eur FROM ledger "
     "WHERE account_type='Expense' GROUP BY 1 ORDER BY 2 DESC"),
    (["revenue", "region"], "Revenue by region",
     "SELECT region, ROUND(SUM(amount_eur),2) AS revenue_eur FROM ledger "
     "WHERE account_type='Revenue' GROUP BY 1 ORDER BY 2 DESC"),
    (["revenue", "month"], "Monthly revenue trend",
     "SELECT month, ROUND(SUM(amount_eur),2) AS revenue_eur FROM ledger "
     "WHERE account_type='Revenue' GROUP BY 1 ORDER BY 1"),
    (["largest", "transaction"], "10 largest expense transactions",
     "SELECT txn_id, txn_date, business_unit, category, counterparty, amount_eur FROM ledger "
     "WHERE account_type='Expense' ORDER BY amount_eur DESC LIMIT 10"),
    (["peer", "revenue"], "Latest quarterly revenue of public peers (SEC)",
     "SELECT company, fiscal_quarter, ROUND(revenue_usd/1e9,2) AS revenue_usd_bn, "
     "ROUND(net_income_usd/1e9,2) AS net_income_usd_bn FROM sec_financials s "
     "WHERE period_end = (SELECT MAX(period_end) FROM sec_financials t WHERE t.ticker=s.ticker) "
     "ORDER BY revenue_usd_bn DESC"),
    (["margin", "peer"], "Net margin of public peers, last 4 quarters (SEC)",
     "SELECT company, ROUND(100.0*SUM(net_income_usd)/SUM(revenue_usd),1) AS net_margin_pct FROM "
     "(SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY period_end DESC) rn FROM sec_financials) "
     "WHERE rn<=4 GROUP BY company ORDER BY 2 DESC"),
]

SUGGESTED_QUESTIONS = [
    "What is the revenue by business unit?",
    "Show revenue, expenses and profit by year",
    "How has monthly cloud spend developed?",
    "Who are our top vendors by spend?",
    "Which region generates the most revenue?",
    "Compare the latest quarterly revenue of our peers",
]


def match_offline(question: str):
    q = question.lower()
    q = q.replace("expenses", "expense").replace("costs", "expense").replace("cost", "expense") \
         .replace("suppliers", "vendor").replace("vendors", "vendor").replace("monthly", "month") \
         .replace("bu ", "business unit ").replace("competitors", "peer").replace("peers", "peer") \
         .replace("annual", "year").replace("yearly", "year").replace("biggest", "largest") \
         .replace("transactions", "transaction").replace("payments", "transaction").replace("spend", "expense")
    best, score = None, 0
    for kws, title, sql in OFFLINE_INTENTS:
        s = sum(k in q for k in kws)
        if s > score or (s == score and best and len(kws) > len(best[0])):
            best, score = (kws, title, sql), s
    return best if score >= 1 else None


# ---------------- SQL safety ----------------
def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip().rstrip(";").strip()


def validate_sql(sql: str) -> str:
    s = sql.strip().rstrip(";")
    if ";" in s:
        raise ValueError("Only a single SQL statement is allowed.")
    if not re.match(r"^\s*(select|with)\b", s, re.I):
        raise ValueError("Only read-only SELECT queries are allowed.")
    if FORBIDDEN.search(s):
        raise ValueError("Query contains a forbidden keyword.")
    return s


def run_sql(con: sqlite3.Connection, sql: str, limit: int = 1000) -> pd.DataFrame:
    s = validate_sql(sql)
    return pd.read_sql_query(f"SELECT * FROM ({s}) LIMIT {limit}", con)


def _offline_summary(title: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "The query returned no rows."
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        return f"{title}: {len(df)} rows returned."
    label = df.columns[0]
    title_words = [w for w in ("revenue", "spend", "expense", "profit", "margin") if w in title.lower()]
    preferred = [c for c in num if c != label and any(w in c for w in title_words)]
    col = (preferred or [c for c in num if c != label] or num)[0]
    top = df.loc[df[col].idxmax()]
    total = df[col].sum()
    if "pct" in col or "margin" in col:
        return f"{title}. Highest {col.replace('_', ' ')}: {top[label]} ({top[col]:.1f}%)."
    return (f"{title}. Total {col.replace('_', ' ')}: {total:,.0f}. "
            f"Highest: {top[label]} with {top[col]:,.0f} ({100 * top[col] / total:.1f}% of total)."
            if total else f"{title}: {len(df)} rows.")


def ask(question: str, con: sqlite3.Connection, use_llm: bool = True) -> QueryResult:
    notes: list[str] = []
    if use_llm and llm.is_configured():
        try:
            gen = llm.generate(question, system=SQL_SYSTEM, max_tokens=500)
            sql = extract_sql(gen["text"])
            try:
                df = run_sql(con, sql)
            except Exception as e:  # one self-correction round
                notes.append(f"First SQL attempt failed ({e}); asked the model to repair it.")
                fix = llm.generate(f"Question: {question}\nYour SQL:\n{sql}\nError: {e}\nReturn corrected SQL.",
                                   system=SQL_SYSTEM, max_tokens=500)
                sql = extract_sql(fix["text"])
                df = run_sql(con, sql)
            table = df.head(25).to_csv(index=False)
            summ = llm.generate(f"Question: {question}\nSQL: {sql}\nResult (CSV, first 25 rows of {len(df)}):\n{table}",
                                system=SUMMARY_SYSTEM, max_tokens=250, temperature=0.2)
            mode = "gemini (cached)" if gen["cached"] and summ["cached"] else "gemini"
            return QueryResult(question, sql, df, summ["text"], mode, notes)
        except llm.LLMUnavailable as e:
            notes.append(f"Gemini unavailable: {e}. Fell back to offline mode.")
        except Exception as e:
            notes.append(f"Gemini answer could not be executed safely ({e}). Fell back to offline mode.")

    hit = match_offline(question)
    if not hit:
        return QueryResult(question, "", pd.DataFrame(),
                           "Offline mode could not map this question. Try one of the suggested questions, "
                           "or add a GEMINI_API_KEY to unlock free-form questions.", "offline", notes)
    _, title, sql = hit
    df = run_sql(con, sql)
    from . import green
    green.record(offline_answers=1)
    return QueryResult(question, sql, df, _offline_summary(title, df), "offline", notes)
